import os
import sys
import json
import random
import warnings
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from WFlib import models
from WFlib.tools import data_processor, evaluator

sys.path.append(os.path.dirname(__file__))
from utils.debug import *

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

parser = argparse.ArgumentParser(description="WFlib")
args = parser.parse_args()


# -- Utilities -- #


def load_data(data_path, extra_data_path=None):
    data, labels = data_processor.load_data(data_path, args.feature, args.seq_len, args.num_tabs)
    if extra_data_path is not None:
        extra_data, extra_labels = data_processor.load_data(extra_data_path, args.feature, args.seq_len, args.num_tabs)
        data = torch.cat([data, extra_data], dim=0)
        labels = torch.cat([labels, extra_labels], dim=0)
    return data, labels


def infinite_iter(loader):
    while True:
        yield from loader


# -- Core Logic -- #


def split_target_by_energy():
    pass


# -- Model Tuning -- #


def adapt_model(model, train_loader, tune_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    known_mask = unknown_mask = None
    pseudo_loader = pesudo_iter = None

    for epoch in range(args.tune_epochs):
        if epoch % args.split_refresh == 0:
            known_mask, unknown_mask = split_target_by_energy(model, tune_loader, tau, device)
            known_data = 


def main():
    device = torch.device(args.device)

    # dataset, log, checkpoint
    dataset_path = os.path.join("./datasets", args.dataset)
    log_path = os.path.join(args.log_path, args.dataset, args.model)
    ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckp_path, exist_ok=True)

    train_path = os.path.join(dataset_path, f"{args.train_file}.npz")
    test_path = os.path.join(dataset_path, f"{args.test_file}.npz")
    tune_path = os.path.join(dataset_path, f"{args.tune_file}.npz")

    # train, test, tune data
    extra_train_path = f"{args.extra_train_file}.npz" if args.use_extra_train_file else None
    extra_test_path = f"{args.extra_test_file}.npz" if args.use_extra_test_file else None
    extra_tune_path = f"{args.extra_tune_file}.npz" if args.use_extra_tune_file else None

    train_datas, train_labels = load_data(train_path, extra_train_path)
    test_datas, test_labels = load_data(test_path, extra_test_path)
    tune_datas, tune_labels = load_data(tune_path, extra_tune_path)

    print(f"\n{'='*20} Configuration {'='*20}")
    print(f"Dataset: {args.dataset},  Model: {args.model},  Device: {device}")
    print(f"Train: {train_datas.shape},  Tune: {tune_datas.shape},  Test: {test_datas.shape}")
    print(f"Source classes: {num_classes},  Unknown label: {args.unknown_label}")
    print(f"{'='*55}\n")

    # train, test, tune dataloader
    train_loader = data_processor.create_dataloader(train_datas, train_labels, args.batch_size, True, args.num_workers)
    tune_loader = data_processor.create_dataloader(
        tune_datas, torch.zero_like(tune_labels), args.batch_size, False, args.num_workers
    )
    test_loader = data_processor.create_dataloader(test_datas, test_labels, args.batch_size, False, args.num_workers)

    # model
    model = eval(f"models.{args.model}")(num_classes)
    model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.load_name}.pth"), map_location="cpu"))
    model.to(device)

    # evaluate before tuning
    # TODO

    # model tuning
    # TODO

    # evaluate after tuning
    # TODO

    # Save model
    model_save_path = os.path.join(ckp_path, f"{args.model_save_name}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Save results to file
    with open(output_file, "w") as result_file:
        json.dump(final_result, result_file, indent=4)
