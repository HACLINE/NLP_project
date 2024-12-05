import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from pathlib import Path

import json
import shutil
import logging
import random
from types import SimpleNamespace
import time
from pprint import pprint
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from sklearn import preprocessing
import wandb
from torch.utils.tensorboard import SummaryWriter

from scrl.manager import TrainingManager
import scrl.utils as utils
import scrl.sampling as sampling

from nltk import word_tokenize

def setup_model(args):
    # setup/load model manager object
    model_dir = Path(args.model_dir)
    if args.fresh and model_dir.exists():
        utils.ask_rmdir(model_dir)
    manager = TrainingManager(model_dir, args)
    if not manager.is_empty():
        manager.load()

    if not (model_dir / "config.json").exists():
        shutil.copy(args.config, model_dir / "config.json")

    encoder = AutoModel.from_pretrained(args.encoder_model_id).to(args.device)
    embedding_size = encoder.state_dict()["embeddings.word_embeddings.weight"].shape[1]
    model = globals()[args.model_name](encoder, embedding_size, args.device, args.model_kwargs).to(args.device)    
    if manager.step > 0:
        print("loading latest model from step", manager.step - 1)
        model.load(model_dir, prefix="latest")
    return manager, model

class BaseModel(nn.Module):
    """
    Base class for all models.

    Basic elements:
    - self.encoder: nn.Module, pretrained encoder
    - self.embedding_size: int, size of the embeddings
    - self.classifier: input encoded embeddings -> log-probabilities over tokens, 1 for keep, 0 for remove
    - self.device: str, device
    - self.networks: list of str, names of the networks in the model
    """
    def __init__(self, encoder, embedding_size, device, kwargs):
        """
        Init.

        args:
        - encoder: nn.Module, pretrained encoder
        - embedding_size: int, size of the embeddings
        - kwargs: dict, additional keyword arguments
        """
        super(BaseModel, self).__init__()
        self.encoder = encoder
        self.embedding_size = embedding_size
        self.classifier = None
        self.networks = ["encoder"]
        
        self.device = device
        self.kwargs = kwargs.copy()
        self.kwargs["embedding_size"] = embedding_size
        self.kwargs["model_name"] = self.__class__.__name__
        kwargs["device"] = device
        self.args = SimpleNamespace(**kwargs)

    def forward(self, x):
        """
        Forward pass.

        args:
        - x: torch.Tensor, shape (batch_size, seq_len)

        return:
        - logits: torch.Tensor, shape (batch_size, seq_len, 2)
        """
        raise NotImplementedError

    def predict(self, texts, tokenizer):
        """
        Predict summaries.

        args:
        - texts: list of str, input texts
        - tokenizer: tokenizer

        return:
        - summaries: list of str, predicted summaries
        """
        raise NotImplementedError
    
    def update(self,
               batch,
               optimizer,
               tokenizer,
               reward_generator,
               manager,
               ):
        """
        Update the model.

        args:
        - batch: dict, batch data
        - optimizer: torch.optim.Optimizer
        - tokenizer: tokenizer
        - reward_generator: reward generator
        - manager: TrainingManager

        return:
        - probs: torch.Tensor, shape (batch_size, seq_len, 2)
        - argmax_summaries: list of str, argmax summaries
        - sample_summaries: list of str, sample summaries
        - argmax_details: dict, reward details, e.g. {'BiEncoderSimilarity': [0.8817998170852661],
 'GaussianCR': [0.9888130446112331]}
        """
        raise NotImplementedError
    
    def save(self, path):
        """
        Save the model.

        args:
        path: str, all networks will be saved to this path
        """
        for name in self.networks:
            if name == "encoder":
                self.encoder.save_pretrained(path / "encoder.bin")
            else:
                torch.save(getattr(self, name).state_dict(), path / f"{name}.bin")
        with open(path / "kwargs.json", "w") as f:
            json.dump(self.kwargs, f)

    def load(self, model_dir, prefix="best"):
        """
        Load the model.

        args:
        - model_dir: str|Path, path to the model directory
        - prefix: str, prefix of the model to load
        """
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        for p in (model_dir / "checkpoints").iterdir():
            if p.name.startswith(f"{prefix}"):
                checkpoint_dir = p
        self.load_checkpoint(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_dir):
        """
        Load the model from a checkpoint directory.

        args:
        - checkpoint_dir: str|Path, path to the checkpoint directory
        """
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        with open(checkpoint_dir / "kwargs.json", "r") as f:
            self.kwargs = json.load(f)

        encoder_path = checkpoint_dir / "encoder.bin"
        self.encoder = AutoModel.from_pretrained(encoder_path).to(self.device)
        embedding_size = self.encoder.state_dict()["embeddings.word_embeddings.weight"].shape[1]
        assert embedding_size == self.embedding_size

        for name in self.networks:
            if name == "encoder":
                continue
            state = torch.load(checkpoint_dir / f"{name}.bin", map_location=self.device)
            getattr(self, name).load_state_dict(state)

    def manager_update(self,
                       manager,
                       loss,
                       a_reward,
                       s_reward,
                       sample_probs,
                       probs,
                       argmax_len,
                       argmax_details
                       ):
        """
        Update the model manager.
        """

        manager.update_metric("time", time.time())
        manager.update_metric("loss", loss.item())
        manager.update_metric("argmax_reward", a_reward)
        manager.update_metric("sample_reward", s_reward)
        manager.update_metric("sample_prob", sample_probs.detach().cpu().numpy().mean())
        manager.update_metric("mean_max_prob", get_mean_max_prob(probs))
        manager.update_metric("label_variance", label_variance(probs))
        manager.update_metric("argmax_len", argmax_len)
        for rname, rvalues in argmax_details.items():
            manager.update_metric(f"reward|{rname}", np.mean(rvalues))

class LinearTokenSelector(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(LinearTokenSelector, self).__init__(encoder, embedding_size, device, kwargs)
        self.classifier = nn.Linear(embedding_size, 2, bias=False)
        self.networks.append("classifier")

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1] # B * S * H
        x = self.classifier(x)
        x = F.log_softmax(x, dim=2)
        return x

    def predict(self, texts, tokenizer):
        input_ids = tokenizer(texts)["input_ids"]
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True
        ).to(self.device)
        logits = self.forward(input_ids)
        argmax_labels = torch.argmax(logits, dim=2)
        return utils.labels_to_summary(input_ids, argmax_labels, tokenizer)

    def update(self,
               batch,
               optimizer,
               tokenizer,
               reward_generator,
               manager,
               ):
        optimizer.zero_grad()

        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in batch["input_ids"]],
            batch_first=True
        ).to(self.device)

        logits = self.forward(input_ids)
        probs = torch.softmax(logits, dim=2)

        argmax_labels = torch.argmax(logits, dim=2).to(self.device)
        argmax_summaries = utils.labels_to_summary(input_ids, argmax_labels, tokenizer)
        argmax_rewards, argmax_details = reward_generator(batch["document"], argmax_summaries)
        a_reward = np.mean(argmax_rewards)

        (sample_probs, sample_summaries, sample_rewards, sample_details,
        sample_labels, sample_data) = sampling.best_of_k_samples(
            self.args, manager, tokenizer, reward_generator,
            input_ids, batch, probs,
            k_samples=self.args.k_samples,
        )
        s_reward = np.mean(sample_rewards)

        if self.args.sample_aggregation == "max":
            loss = (a_reward - s_reward) * sample_probs.sum(1).mean()
        else:
            loss = 0.
            for sample_probs_i, s_rewards_i, a_rewards_i in zip(sample_data["probs"], sample_data["rewards"], sample_rewards):
                s_reward_i = np.mean(s_rewards_i)
                a_reward_i = np.mean(a_rewards_i)
                loss_i = (a_reward_i - s_reward_i) * sample_probs_i.sum(1).mean()
                loss += loss_i
            loss /= len(sample_data["rewards"])

        if self.args.sample_aggregation == "mean" or a_reward != s_reward:
            # not updating model if no reward difference, in case of single sample
            loss.backward()
            optimizer.step()

        argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

        self.manager_update(manager,
                            loss,
                            a_reward,
                            s_reward,
                            sample_probs,
                            probs,
                            argmax_len,
                            argmax_details
                            )

        return probs, argmax_summaries, sample_summaries, argmax_details    

def label_variance(probs):
    # batch, seq, 2
    variances = []
    for i in range(probs.size(0)):
        distrib = probs[i, :, 0]
        var = torch.var(distrib)
        variances.append(var)
    return var.mean().item()

def get_mean_max_prob(probs):
    return probs.max(dim=2).values.mean().item()

def load_model(model_dir, device="cuda", prefix="best"):
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    for p in (model_dir / "checkpoints").iterdir():
        if p.name.startswith(f"{prefix}"):
            checkpoint_dir = p
    return load_checkpoint(checkpoint_dir, device=device)

def load_checkpoint(checkpoint_dir, device="cuda"):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    with open(checkpoint_dir / "kwargs.json", "r") as f:
        kwargs = json.load(f)

    encoder_path = checkpoint_dir / "encoder.bin"
    encoder = AutoModel.from_pretrained(encoder_path).to(device)
    embedding_size = encoder.state_dict()["embeddings.word_embeddings.weight"].shape[1]

    model = globals()[kwargs["model_name"]](encoder, embedding_size, device, kwargs).to(device)
    model.load_checkpoint(checkpoint_dir)
    return model

