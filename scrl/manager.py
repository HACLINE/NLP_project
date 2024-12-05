import argparse
import shutil
import logging
import random
import time
from pprint import pprint
from collections import defaultdict
from pathlib import Path

import scrl.utils as utils

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from sklearn import preprocessing
import wandb
from torch.utils.tensorboard import SummaryWriter

class TrainingManager:
    """
    Object for saving/loading model checkpoints and for tracking and saving
    metrics measured during training, e.g. loss, rewards.

    The following directory struture is build around one training run:

    dir/
        val_scores.json
        checkpoints/
            latest-model-500/
                classifier.bin
                encoder.bin
            best-model-200/
                [...]
        series/
            loss.npy
            [...]
        totals/
            loss.npy
            [...]
    """
    def __init__(self, dir, args):
        self.step = 0
        self.total_seconds = 0
        self.start_time = None
        self.series = defaultdict(list)
        self.totals = defaultdict(float)
        self.dir = dir
        dir.mkdir(exist_ok=True)
        for subdir_name in ("checkpoints", "series", "totals", "tb"):
            (dir / subdir_name).mkdir(exist_ok=True)

        if args.use_tb:
            self.tb_writer = SummaryWriter(str(dir / "tb"))
        else:
            self.tb_writer = None

        self.wandb_run_id = None
        if args.use_wandb:
            wandb_id_file = dir / "wandb_run_id.json"
            if wandb_id_file.exists():
                self.wandb_run_id = utils.read_json(wandb_id_file)["id"]
            
            wandb.init(project=args.wandb["project"], config=args, resume="allow", id=self.wandb_run_id, dir=str(dir / "wandb"), name=args.wandb["name"])
            self.wandb_run_id = wandb.run.id

            utils.write_json({"id": self.wandb_run_id}, wandb_id_file)

    def start_clock(self):
        self.start_time = time.time() - self.total_seconds

    def load(self):
        # load tracked data, e.g. loss, rewards etc.
        for p in (self.dir / "series").iterdir():
            k = p.name.split(".npy")[0]
            self.series[k] = list(utils.load_numpy(p))
        for p in (self.dir / "totals").iterdir():
            k = p.name.split(".npy")[0]
            self.totals[k] = utils.load_numpy(p)
        # read latest training step
        latest_model_dir = self.find_old_model("latest-model")
        self.total_seconds = utils.read_json(self.dir / "time.json")["total_seconds"]
        last_step = int(latest_model_dir.name.split("-")[-1])
        self.step = last_step + 1

    def update_metric(self, key, value):
        self.totals[key] += value
        self.series[key].append(value)
        
        if self.tb_writer:
            self.tb_writer.add_scalar(key, value, self.step)

        if self.wandb_run_id:
            wandb.log({key: value}, step=self.step)
    
    def update_metrics(self, metrics):
        for k, v in metrics.items():
            self.update_metric(k, v)

    def mean_metric(self, key):
        return self.totals[key] / (self.step + 1)

    def save_latest_model(self, model, checkpoint_id):
        self.save_model(model, checkpoint_id, prefix="latest-model")

    def save_model(self, model, checkpoint_id, prefix):
        old_model_dir = self.find_old_model(prefix)
        model_dir = self.dir / "checkpoints" / f"{prefix}-{checkpoint_id}"
        model_dir.mkdir()
        model.save(model_dir)
        if old_model_dir:
            shutil.rmtree(old_model_dir)

    def find_old_model(self, prefix):
        model_path = None
        for p in (self.dir / "checkpoints").iterdir():
            if p.name.startswith(f"{prefix}"):
                model_path = p
        return model_path

    def is_empty(self):
        latest_model_dir = self.find_old_model("latest-model")
        return latest_model_dir is None

    def save_data(self):
        for k, v in self.series.items():
            utils.save_numpy(v, self.dir / "series" / f"{k}.npy")
        for k, v in self.totals.items():
            utils.save_numpy(v, self.dir / "totals" / f"{k}.npy")
        utils.write_json({
            "step": self.step,
            "total_seconds": self.total_seconds
        }, self.dir / "time.json")
