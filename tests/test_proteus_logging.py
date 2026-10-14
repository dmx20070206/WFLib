"""Logging aggregation, report serialization, and current Proteus integration."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from exp import log, proteus
from WFlib.models import DMX


class ProteusLoggingTest(unittest.TestCase):
    def tearDown(self):
        for handler in list(log._LOGGER.handlers):
            handler.close()
            log._LOGGER.removeHandler(handler)

    @staticmethod
    def bank():
        return SimpleNamespace(
            support_maps=torch.zeros(3, 512, 2),
            support_labels=torch.tensor([0, 0, 1]),
            descriptors=[torch.zeros(4, 512), torch.zeros(2, 512)],
            variances=torch.tensor([0.25, 0.0]),
            weights=torch.tensor([0.8, 0.5]),
        )

    def test_weighted_means_detach_and_handle_optional_metrics(self):
        metrics = log.EpochMetrics()
        loss = torch.tensor(2.0, requires_grad=True)
        metrics.update(weight=3, loss=loss, target=4)
        metrics.update(weight=1, loss=torch.tensor(6.0))
        with patch.object(log, "log_epoch") as output:
            values = metrics.log(3, 1, 2, 0.75, acc_scope="held-out target")
        self.assertEqual(values["loss"], 3.0)
        self.assertEqual(values["target"], 4.0)
        self.assertEqual(values["batches"], 2)
        self.assertEqual(values["total_weight"], 4)
        self.assertIsNone(loss.grad)
        self.assertIsInstance(metrics.sums["loss"], float)
        output.assert_called_once_with(3, 1, 2, values, 0.75, "held-out target")

    def test_batch_means_empty_epoch_and_zero_weight(self):
        metrics = log.EpochMetrics()
        with patch.object(log, "log_epoch"):
            self.assertIsNone(metrics.log(2, 1, 1)["loss"])
            metrics.update(weight=0, loss=100)
            metrics.update(loss=2)
            metrics.update(loss=4)
            self.assertEqual(metrics.log(2, 1, 1)["loss"], 3)
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                metrics.update(weight=weight, loss=1)

    def test_bank_and_checkpoint_logging_stays_in_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "run.log"
            console = io.StringIO()
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                log.configure_logging(path, {"ot_eta": 0.1}, "cpu")
                summary = log.log_target_bank(self.bank())
                log.log_stage2_checkpoint("translator.pth")
                log.log_stage2_checkpoint("translator.pth", loaded=True)
            self.assertEqual(console.getvalue(), "")
            self.assertEqual(summary["support_samples"], 3)
            self.assertEqual(summary["per_class"]["0"]["descriptor_count"], 4)
            self.assertEqual(summary["per_class"]["1"]["support_count"], 1)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Target support bank", text)
            self.assertIn("checkpoint saved", text)
            self.assertIn("loaded (training skipped)", text)
            self.assertIn("map_reg_weight", text)

    def test_report_serializes_numpy_tensors_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(log_path=Path(directory), dataset="tiny",
                                   model="DMX", result_file="run", device=torch.device("cpu"))
            bank = self.bank()
            bank.variances[0] = float("nan")
            path = log.save_run_report(
                args, {"Accuracy": np.float32(0.75)},
                {"Accuracy": torch.tensor(0.5)}, bank,
                np.array([1, 3, 5]), Path("stage2.pth"),
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path, Path(directory) / "tiny" / "DMX" / "run.json")
            self.assertEqual(report["result"]["Accuracy"], 0.75)
            self.assertEqual(report["baseline"]["Accuracy"], 0.5)
            self.assertEqual(report["support_indices"], [1, 3, 5])
            self.assertEqual(report["stage2_checkpoint"], "stage2.pth")
            self.assertEqual(report["config"]["device"], "cpu")
            self.assertIsNone(report["target_bank"]["per_class"]["0"]["variance"])
            self.assertNotIn("descriptors", report["target_bank"])

    def test_main_trains_and_reloads_stage2_with_all_logging_calls(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(7)
                labels = torch.tensor([0, 0, 1, 1])
                loader = proteus.make_loader(torch.rand(4, 1, 2, 180), labels, 3, False, 0)
                data = proteus.AdaptationData(loader, loader, loader, np.arange(4), 2)
                checkpoint_dir = Path(directory) / "checkpoints" / "tiny" / "DMX"
                checkpoint_dir.mkdir(parents=True)
                torch.save(DMX(2).state_dict(), checkpoint_dir / "max_f1.pth")
                argv = ["--dataset", "tiny", "--checkpoints", str(Path(directory) / "checkpoints"),
                        "--log_path", str(Path(directory) / "logs"), "--num_workers", "0",
                        "--batch_size", "3", "--stage2_epochs", "1", "--stage3_epochs", "1",
                        "--ot_iterations", "2", "--ot_max_descriptors", "4"]
                with patch.object(proteus, "prepare_data", return_value=data), contextlib.redirect_stdout(io.StringIO()):
                    result = proteus.main(argv)
                    with patch.object(proteus, "train_stage2") as train_stage2:
                        proteus.main(argv + ["--load_stage2"])
                        train_stage2.assert_not_called()
                report_path = Path(directory) / "logs" / "tiny" / "DMX" / "proteus.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertIn("Accuracy", result)
                self.assertTrue(Path(report["stage2_checkpoint"]).is_file())
                self.assertTrue((checkpoint_dir / "proteus.pth").is_file())
                self.assertEqual(report["support_indices"], [0, 1, 2, 3])
                text = report_path.with_suffix(".log").read_text(encoding="utf-8")
                self.assertIn("[Stage 2][Epoch 1/1] loss=", text)
                self.assertIn("[Stage 3][Epoch 1/1] loss=", text)
                self.assertIn("[Final] acc=", text)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
