import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from WFlib.models import ProjectionHead, SwallowEncoder


SWALLOW_SLOT_LEN = 1000
SWALLOW_BACKBONE = "resnet18"
SWALLOW_MOMENTUM = 0.996
SWALLOW_PROJECTION_HIDDEN = 512
SWALLOW_PROJECTION_SIZE = 128
SWALLOW_MULTI_TIMES = 6
SWALLOW_N_VIEWS = 2
SWALLOW_WEIGHT_DECAY = 4e-4
SWALLOW_MOMENTUM_OPT = 0.9


def median_of_non_zero(values):
    non_zero_values = [value for value in values if value != 0]
    if not non_zero_values:
        return None

    non_zero_values.sort()
    value_count = len(non_zero_values)
    middle = value_count // 2
    if value_count % 2 == 1:
        return non_zero_values[middle]
    return (non_zero_values[middle - 1] + non_zero_values[middle]) / 2


class SwallowAugmentor:
    def __init__(self):
        self.handshake_packet_sum = 20
        self.remain_zero_prob = 0.5
        self.alpha_upload = 0.7
        self.alpha_download = 0.7
        self.num_window_avg = 20
        self.shrink_ratio_max = 0.7
        self.shrink_ratio_min = 0.3
        self.shrink_prob = 0.6
        self.insert_prob = 0.1
        self.remove_prob = 0.7
        self.increase_ratio_max = 0.7
        self.increase_ratio_min = 0.3
        self.increase_prob = 0.6

    def trace_fluctuation(self, cif, load_time, slot_duration):
        packet_sum = 0
        upload_value = cif[0]
        download_value = cif[1]
        upload_value_ori = upload_value.copy()
        download_value_ori = download_value.copy()

        for index in range(len(upload_value)):
            if index * slot_duration >= load_time:
                break

            time_slot_upload = upload_value[index]
            time_slot_download = download_value[index]
            packet_sum += time_slot_upload + time_slot_download
            if packet_sum <= self.handshake_packet_sum:
                continue

            if time_slot_upload == 0:
                if random.random() > self.remain_zero_prob:
                    start = max(0, index - self.num_window_avg)
                    end = min(len(upload_value), index + self.num_window_avg + 1)
                    if end - start >= self.num_window_avg:
                        time_slot_upload = int(np.mean(upload_value[start:end]))
            else:
                scale = 1 - self.alpha_upload * random.random()
                if random.random() > 0.5:
                    scale = 1 + self.alpha_upload * random.random()
                time_slot_upload = int(scale * time_slot_upload)

            if time_slot_download == 0:
                if random.random() > self.remain_zero_prob:
                    start = max(0, index - self.num_window_avg)
                    end = min(len(download_value), index + self.num_window_avg + 1)
                    if end - start >= self.num_window_avg:
                        time_slot_download = int(np.mean(download_value[start:end]))
            else:
                scale = 1 - self.alpha_download * random.random()
                if random.random() > 0.5:
                    scale = 1 + self.alpha_download * random.random()
                time_slot_download = int(scale * time_slot_download)

            upload_value_ori[index] = time_slot_upload
            download_value_ori[index] = time_slot_download

        return np.vstack([np.asarray(upload_value_ori), np.asarray(download_value_ori)])

    def trace_aggregation(self, cif, load_time, slot_duration):
        packet_sum = 0
        upload_value = cif[0]
        download_value = cif[1]
        time_slot_len = len(upload_value)
        real_upload_value = []
        real_download_value = []

        index = 0
        while index * slot_duration <= load_time and index < time_slot_len:
            time_slot_upload = upload_value[index]
            time_slot_download = download_value[index]
            packet_sum += time_slot_upload + time_slot_download

            if packet_sum <= self.handshake_packet_sum:
                index += 1
                continue

            if random.random() <= self.insert_prob:
                start = max(0, index - self.num_window_avg)
                end = min(len(download_value), index + self.num_window_avg + 1)
                if end - start >= self.num_window_avg:
                    time_slot_upload = int(np.mean(upload_value[start:end]))
                    time_slot_download = int(np.mean(download_value[start:end]))
                else:
                    time_slot_upload = median_of_non_zero(upload_value) or 0
                    time_slot_download = median_of_non_zero(download_value) or 0
                real_upload_value.append(time_slot_upload)
                real_download_value.append(time_slot_download)
            else:
                if random.random() >= self.shrink_prob:
                    time_slot_upload = int(
                        time_slot_upload * random.uniform(self.shrink_ratio_min, self.shrink_ratio_max)
                    )
                    time_slot_download = int(
                        time_slot_download * random.uniform(self.shrink_ratio_min, self.shrink_ratio_max)
                    )
                    real_upload_value.append(time_slot_upload)
                    real_download_value.append(time_slot_download)
                index += 1

        for _ in range(time_slot_len - len(real_upload_value)):
            real_upload_value.append(0)
            real_download_value.append(0)

        return np.vstack([np.asarray(real_upload_value), np.asarray(real_download_value)])

    def trace_flatten(self, cif, load_time, slot_duration):
        packet_sum = 0
        upload_value = cif[0]
        download_value = cif[1]
        real_upload_value = []
        real_download_value = []

        for index in range(len(upload_value)):
            time_slot_upload = upload_value[index]
            time_slot_download = download_value[index]
            packet_sum += time_slot_upload + time_slot_download

            if packet_sum <= self.handshake_packet_sum:
                continue
            if index * slot_duration >= load_time:
                break
            if random.random() > self.remove_prob:
                continue

            if random.random() > self.increase_prob:
                time_slot_upload = int(
                    time_slot_upload * (1 + random.uniform(self.increase_ratio_min, self.increase_ratio_max))
                )
                time_slot_download = int(
                    time_slot_download * (1 + random.uniform(self.increase_ratio_min, self.increase_ratio_max))
                )

            real_upload_value.append(time_slot_upload)
            real_download_value.append(time_slot_download)

        for _ in range(len(upload_value) - len(real_upload_value)):
            real_upload_value.append(0)
            real_download_value.append(0)

        return np.vstack([np.asarray(real_upload_value), np.asarray(real_download_value)])

    def augment(self, cif, load_time, slot_duration):
        selected_number = random.choice([0, 1, 2])
        if selected_number == 0:
            return self.trace_fluctuation(cif, load_time, slot_duration)
        if selected_number == 1:
            return self.trace_aggregation(cif, load_time, slot_duration)
        return self.trace_flatten(cif, load_time, slot_duration)


class SwallowPreTrainDataset(Dataset):
    def __init__(self, features, labels, load_times, slot_durations, augmentor, n_views):
        self.features = features
        self.labels = labels
        self.load_times = load_times
        self.slot_durations = slot_durations
        self.augmentor = augmentor
        self.n_views = n_views

    def __getitem__(self, index):
        views = [
            np.asarray(
                self.augmentor.augment(
                    self.features[index].copy(),
                    self.load_times[index],
                    self.slot_durations[index],
                ),
                dtype=np.float32,
            )
            for _ in range(self.n_views)
        ]
        return views, int(self.labels[index])

    def __len__(self):
        return len(self.labels)


class SwallowBYOLTrainer:
    def __init__(
        self,
        online_encoder,
        target_encoder,
        online_projector,
        target_projector,
        predictor,
        optimizer,
        device,
        max_epochs,
        momentum,
    ):
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.online_projector = online_projector
        self.target_projector = target_projector
        self.predictor = predictor
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.momentum = momentum

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initialize_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = target_param.data * self.momentum + online_param.data * (1.0 - self.momentum)
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = target_param.data * self.momentum + online_param.data * (1.0 - self.momentum)

    def train(self, train_loader):
        self.initialize_target()
        for epoch_counter in range(self.max_epochs):
            self.online_encoder.train()
            self.online_projector.train()
            self.predictor.train()
            total_loss = 0.0
            total_batch = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch_counter + 1:03d}/{self.max_epochs} [Pretrain]", leave=False, dynamic_ncols=True)
            for (view_1, view_2), _ in progress:
                view_1 = view_1.to(self.device).float().unsqueeze(1)
                view_2 = view_2.to(self.device).float().unsqueeze(1)

                proj_1 = self.online_projector(self.online_encoder(view_1))
                proj_2 = self.online_projector(self.online_encoder(view_2))
                pred_1 = self.predictor(proj_1)
                pred_2 = self.predictor(proj_2)

                with torch.no_grad():
                    self.target_projector.eval()
                    target_1 = self.target_projector(self.target_encoder(view_1))
                    target_2 = self.target_projector(self.target_encoder(view_2))

                loss = self.regression_loss(pred_1, target_2) + self.regression_loss(pred_2, target_1)
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_target()

                total_loss += float(loss.detach().cpu().item())
                total_batch += 1
                progress.set_postfix({"Loss": f"{loss.item():.4f}"})

            mean_loss = total_loss / max(total_batch, 1)
            print(f"Epoch {epoch_counter + 1:03d} | Pretrain Loss: {mean_loss:.4f}")


def sequence_to_cif(sequence, slot_len=SWALLOW_SLOT_LEN):
    packets = np.asarray(sequence, dtype=np.float64)
    packets = packets[packets != 0]
    feature = np.zeros((2, slot_len), dtype=np.float32)
    if packets.size == 0:
        return feature, 0.0, 0.02

    packet_times = np.abs(packets)
    packet_times = packet_times - packet_times[0]
    load_time = float(packet_times[-1]) if packet_times.size else 0.0
    slot_duration = 3 * load_time / slot_len if load_time > 0 else 0.02
    slot_duration = min(max(slot_duration, 0.02), 0.08)

    for packet, packet_time in zip(packets, packet_times):
        slot_index = int(packet_time // slot_duration) if slot_duration > 0 else 0
        if slot_index >= slot_len:
            break
        feature[0 if packet > 0 else 1, slot_index] += 1

    return feature, load_time, slot_duration


def build_cif_features(sequences, show_progress=False, desc="Extracting CIF"):
    features = []
    load_times = []
    slot_durations = []

    iterator = sequences
    if show_progress:
        iterator = tqdm(sequences, desc=desc, leave=False, dynamic_ncols=True)

    for sequence in iterator:
        feature, load_time, slot_duration = sequence_to_cif(sequence)
        features.append(feature)
        load_times.append(load_time)
        slot_durations.append(slot_duration)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(load_times, dtype=np.float32),
        np.asarray(slot_durations, dtype=np.float32),
    )


def load_npz_to_cif(path, show_progress=False, desc=None):
    data = np.load(path)
    features, load_times, slot_durations = build_cif_features(
        data["X"],
        show_progress=show_progress,
        desc=desc or f"Extracting CIF: {path}",
    )
    return features, data["y"].astype(np.int64), load_times, slot_durations


def concat_cif_parts(parts):
    features = np.concatenate([part[0] for part in parts], axis=0)
    labels = np.concatenate([part[1] for part in parts], axis=0)
    load_times = np.concatenate([part[2] for part in parts], axis=0)
    slot_durations = np.concatenate([part[3] for part in parts], axis=0)
    return features, labels, load_times, slot_durations


def build_pretrain_loader(file_paths, batch_size, num_workers):
    data_parts = [load_npz_to_cif(path, show_progress=True) for path in file_paths]
    features, labels, load_times, slot_durations = concat_cif_parts(data_parts)
    augmentor = SwallowAugmentor()
    dataset = SwallowPreTrainDataset(
        features,
        labels,
        load_times,
        slot_durations,
        augmentor,
        SWALLOW_N_VIEWS,
    )
    train_dataset = ConcatDataset([dataset for _ in range(SWALLOW_MULTI_TIMES)])
    effective_batch_size = min(batch_size, len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=len(train_dataset) > effective_batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, features.shape[0]


def build_pretrain_components(device, learning_rate=None):
    online_encoder = SwallowEncoder(SWALLOW_BACKBONE).to(device)
    target_encoder = SwallowEncoder(SWALLOW_BACKBONE).to(device)
    online_projector = ProjectionHead(
        in_channels=online_encoder.out_dim,
        mlp_hidden_size=SWALLOW_PROJECTION_HIDDEN,
        projection_size=SWALLOW_PROJECTION_SIZE,
    ).to(device)
    target_projector = ProjectionHead(
        in_channels=target_encoder.out_dim,
        mlp_hidden_size=SWALLOW_PROJECTION_HIDDEN,
        projection_size=SWALLOW_PROJECTION_SIZE,
    ).to(device)
    predictor = ProjectionHead(
        in_channels=SWALLOW_PROJECTION_SIZE,
        mlp_hidden_size=SWALLOW_PROJECTION_HIDDEN,
        projection_size=SWALLOW_PROJECTION_SIZE,
    ).to(device)
    optimizer = torch.optim.SGD(
        list(online_encoder.parameters())
        + list(online_projector.parameters())
        + list(predictor.parameters()),
        lr=learning_rate if learning_rate is not None else 0.03,
        momentum=SWALLOW_MOMENTUM_OPT,
        weight_decay=SWALLOW_WEIGHT_DECAY,
    )
    return online_encoder, target_encoder, online_projector, target_projector, predictor, optimizer


def run_pretrain(file_paths, out_file, device, train_epochs, batch_size, num_workers, learning_rate=None):
    train_loader, sample_count = build_pretrain_loader(file_paths, batch_size, num_workers)
    print(f"Pretrain: X={sample_count}")
    online_encoder, target_encoder, online_projector, target_projector, predictor, optimizer = build_pretrain_components(
        device,
        learning_rate,
    )
    trainer = SwallowBYOLTrainer(
        online_encoder=online_encoder,
        target_encoder=target_encoder,
        online_projector=online_projector,
        target_projector=target_projector,
        predictor=predictor,
        optimizer=optimizer,
        device=device,
        max_epochs=train_epochs,
        momentum=SWALLOW_MOMENTUM,
    )
    trainer.train(train_loader)
    torch.save(
        {
            "encoder_state_dict": online_encoder.state_dict(),
            "projector_state_dict": online_projector.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
        },
        out_file,
    )


def load_pretrained_encoder(model, checkpoint_file, map_location="cpu"):
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        return model.encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and all(key.startswith("encoder.") or key.startswith("fc.") for key in checkpoint.keys()):
        return model.load_state_dict(checkpoint, strict=False)
    raise ValueError(f"Unsupported Swallow checkpoint format: {checkpoint_file}")