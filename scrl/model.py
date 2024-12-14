import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from pathlib import Path

import json
import shutil
import logging
import copy
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
from torch.autograd import Variable
from nltk import word_tokenize
import scrl.DATA as DATA
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

    # check kwargs
    if not (checkpoint_dir / "kwargs.json").exists():
        kwargs = {"model_name": "LinearTokenSelector"}
    else:
        with open(checkpoint_dir / "kwargs.json", "r") as f:
            kwargs = json.load(f)

    encoder_path = checkpoint_dir / "encoder.bin"
    encoder = AutoModel.from_pretrained(encoder_path).to(device)
    embedding_size = encoder.state_dict()["embeddings.word_embeddings.weight"].shape[1]

    model = globals()[kwargs["model_name"]](encoder, embedding_size, device, kwargs).to(device)
    model.load_checkpoint(checkpoint_dir)
    return model

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

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
        self.is_encoder = ["encoder"]
        
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
        input_ids = tokenizer(texts)["input_ids"]
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True
        ).to(self.device)
        logits = self.forward(input_ids)
        argmax_labels = torch.argmax(logits, dim=2)
        # print(argmax_labels)
        return utils.labels_to_summary(input_ids, argmax_labels, tokenizer)
    
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
        print(f"Saving model to {path}")
        print(f"kwargs: {self.kwargs}")
        print(f"networks: {self.networks}")
        print(f"is_encoder: {self.is_encoder}")
        for name in self.networks:
            if name in self.is_encoder:
                getattr(self, name).save_pretrained(path / f"{name}.bin")
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

        if not (checkpoint_dir / "kwargs.json").exists():
            self.kwargs = {"model_name": "LinearTokenSelector"}
        else:
            with open(checkpoint_dir / "kwargs.json", "r") as f:
                self.kwargs = json.load(f)

        for name in self.is_encoder:
            encoder_path = checkpoint_dir / f"{name}.bin"
            setattr(self, name, AutoModel.from_pretrained(encoder_path).to(self.device))
            embedding_size = getattr(self, name).state_dict()["embeddings.word_embeddings.weight"].shape[1]
            assert embedding_size == self.embedding_size

        for name in self.networks:
            if name in self.is_encoder:
                continue
            state = torch.load(checkpoint_dir / f"{name}.bin", map_location=self.device)
            for key in list(state.keys()):
                if "classifier." in key:
                    new_key = key.split(".", 1)[1]
                    state[new_key] = state.pop(key)
                    print(f"Renaming {key} to {new_key}")
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

class TransformerTokenSelector(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(TransformerTokenSelector, self).__init__(encoder, embedding_size, device, kwargs)
        self.classifier = nn.Sequential(
            nn.TransformerEncoderLayer(embedding_size, kwargs["num_heads"]),
            nn.Linear(embedding_size, 2)
        )
        self.networks.append("classifier")

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1] # B * S * H
        x = self.classifier(x)
        x = F.log_softmax(x, dim=2)
        return x

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


class ActorCritic(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(ActorCritic, self).__init__(encoder, embedding_size, device, kwargs)
        self.actor = nn.Sequential(
            nn.TransformerEncoderLayer(embedding_size, 2),
            nn.Linear(embedding_size, 2)
        )
        self.critic_encoder = copy.deepcopy(encoder)
        self.critic = nn.Sequential(
            nn.TransformerEncoderLayer(embedding_size, 2),
            nn.Linear(embedding_size, 1),
            Squeeze(-1),
            nn.AdaptiveAvgPool1d(1),
            Squeeze(-1)
        )

        self.target_critic_encoder = copy.deepcopy(self.critic_encoder)
        self.target_critic = copy.deepcopy(self.critic)

        self.networks.extend(["actor", "critic", "target_critic", "critic_encoder", "target_critic_encoder"])
        self.is_encoder.extend(["critic_encoder", "target_critic_encoder"])

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]
        actor_logits = self.actor(x)
        return F.log_softmax(actor_logits, dim=2)
    
    def soft_update(self):
        self.target_critic_encoder.load_state_dict(utils.soft_update_dict(self.target_critic_encoder.state_dict(), self.critic_encoder.state_dict(), self.args.tau))
        self.target_critic.load_state_dict(utils.soft_update_dict(self.target_critic.state_dict(), self.critic.state_dict(), self.args.tau))

    def update(self,
               batch,
               optimizer,
               tokenizer,
               reward_generator,
               manager,
               ):
        for _ in range(self.args.explore_rounds):
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

            argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

            # Extend batch
            prob_batches, summary_batches, reward_batches, detail_batches, label_batches, ids_batches = sampling.k_samples(
                self.args, manager, tokenizer, reward_generator,
                input_ids, batch, probs,
                k_samples=self.args.explore_num,
                return_all=True
            )
            target_critics_batch = [self.target_critic(self.target_critic_encoder(ids, output_hidden_states=True)["hidden_states"][-1]).flatten().detach() for ids in ids_batches]

            # Update critic
            critic_input = self.critic_encoder(input_ids, output_hidden_states=True)["hidden_states"][-1]
            current_critics = self.critic(critic_input)
            target_critics = torch.stack(target_critics_batch).mean(dim=0)
            rewards = torch.tensor(reward_batches, dtype=torch.float).to(self.device)
            rewards = torch.mean(rewards, dim=0)
            target_critics = target_critics * self.args.gamma + rewards
            critic_loss = F.mse_loss(current_critics, target_critics)
            loss = critic_loss

            # Update actor
            current_critics = current_critics.detach()
            actor_loss = 0.
            entropy = 0.
            for sample_probs_i, rewards_i, target_critic_i in zip(prob_batches, reward_batches, target_critics_batch):
                rewards_i = torch.tensor(rewards_i, dtype=torch.float).to(self.device).detach()
                target_critic_i = target_critic_i.detach()
                advantage = rewards_i + self.args.gamma * target_critic_i - current_critics
                sample_probs_i = sample_probs_i.mean(dim=1)
                actor_loss_i = - (sample_probs_i * advantage).mean()
                entropy += - sample_probs_i.mean()
                actor_loss += actor_loss_i
            actor_loss = actor_loss / len(target_critics_batch)
            entropy = entropy / len(target_critics_batch)
            loss += actor_loss - self.args.entropy_weight * entropy

            loss.backward()
            optimizer.step()

            self.soft_update()

            metrics = {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item()
            }
            manager.update_metrics(metrics)


            self.manager_update(manager,
                                loss,
                                a_reward,
                                s_reward,
                                sample_probs,
                                probs,
                                argmax_len,
                                argmax_details
                                )
            
            batch["document"] = argmax_summaries
            batch["input_ids"] = tokenizer(argmax_summaries)["input_ids"]
            # print(f"Round: {_}, Batch: {batch}")

        return probs, argmax_summaries, sample_summaries, argmax_details    

    def predict(self, texts, tokenizer):
        for i in range(self.args.exploit_num):
            input_ids = tokenizer(texts)["input_ids"]
            input_ids = pad_sequence(
                [torch.tensor(ids) for ids in input_ids], batch_first=True
            ).to(self.device)
            logits = self.forward(input_ids)
            argmax_labels = torch.argmax(logits, dim=2)
            texts = utils.labels_to_summary(input_ids, argmax_labels, tokenizer)
        return texts

class LinearQ(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(LinearQ, self).__init__(encoder, embedding_size, device, kwargs)
        self.q_network = nn.Linear(embedding_size, 2)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.networks.extend(["q_network", "target_q_network"])

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]  # B * S * H
        q_values = self.q_network(x)
        return q_values
    def forward_target(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]
        q_values = self.target_q_network(x)
        return q_values
    def soft_update(self):
        self.target_q_network.load_state_dict(utils.soft_update_dict(self.target_q_network.state_dict(), self.q_network.state_dict(), self.args.tau))

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

        q_values = self.forward(input_ids)
        probs = torch.softmax(q_values, dim=2)

        argmax_labels = torch.argmax(q_values, dim=2).to(self.device)
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

        argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

        # Update Q network
        with torch.no_grad():
            target_q_values = self.forward_target(float(input_ids))
            rewards = torch.tensor(sample_rewards, dtype=torch.float).to(self.device)
            rewards = torch.mean(rewards, dim=0)
            target_q_values = rewards + self.args.gamma * target_q_values

        q_loss = F.mse_loss(q_values, target_q_values)
        loss = q_loss

        loss.backward()
        optimizer.step()

        self.soft_update()

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
    
class TransformerQ(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(TransformerQ, self).__init__(encoder, embedding_size, device, kwargs)
        self.q_network = nn.Sequential(
            nn.TransformerEncoderLayer(embedding_size, 2),
            nn.TransformerEncoderLayer(embedding_size, 2),
            nn.Linear(embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.target_q_network = copy.deepcopy(self.q_network)
        self.networks.extend(["q_network", "target_q_network"])

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]  # B * S * H
        q_values = self.q_network(x)
        return q_values
    def forward_target(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]
        q_values = self.target_q_network(x)
        return q_values
    def soft_update(self):
        self.target_q_network.load_state_dict(utils.soft_update_dict(self.target_q_network.state_dict(), self.q_network.state_dict(), self.args.tau))

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

        q_values = self.forward(input_ids)
        probs = torch.softmax(q_values, dim=2)

        argmax_labels = torch.argmax(q_values, dim=2).to(self.device)
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

        argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

        # Update Q network
        with torch.no_grad():
            target_q_values = self.forward_target(input_ids)
            rewards = torch.tensor(sample_rewards, dtype=torch.float).to(self.device)
            rewards = torch.mean(rewards, dim=0)
            target_q_values = rewards + self.args.gamma * target_q_values

        q_loss = F.mse_loss(q_values, target_q_values)
        loss = q_loss

        loss.backward()
        optimizer.step()

        self.soft_update()

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
class HybridQ(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(HybridQ, self).__init__(encoder, embedding_size, device, kwargs)
        self.q_network = nn.Linear(embedding_size, 2)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.networks.extend(["q_network", "target_q_network"])

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]  # B * S * H
        q_values = self.q_network(x)
        return q_values
    def forward_target(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]
        q_values = self.target_q_network(x)
        return q_values

    def soft_update(self):
        self.target_q_network.load_state_dict(utils.soft_update_dict(self.target_q_network.state_dict(), self.q_network.state_dict(), self.args.tau))

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

        q_values = self.forward(input_ids)
        probs = torch.softmax(q_values, dim=2)

        argmax_labels = torch.argmax(q_values, dim=2).to(self.device)
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

        argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

        # Update Q network
        with torch.no_grad():
            target_q_values = self.forward_target(input_ids)
            rewards = torch.tensor(sample_rewards, dtype=torch.float).to(self.device)
            rewards = torch.mean(rewards, dim=0)
            target_q_values = rewards + self.args.gamma * target_q_values

        q_loss = F.mse_loss(q_values, target_q_values)

        if self.args.sample_aggregation == "max":
            loss1 = (a_reward - s_reward) * sample_probs.sum(1).mean()
        else:
            loss1 = 0.
            for sample_probs_i, s_rewards_i, a_rewards_i in zip(sample_data["probs"], sample_data["rewards"], sample_rewards):
                s_reward_i = np.mean(s_rewards_i)
                a_reward_i = np.mean(a_rewards_i)
                loss_i = (a_reward_i - s_reward_i) * sample_probs_i.sum(1).mean()
                loss1 += loss_i
            loss1 /= len(sample_data["rewards"])
        loss = q_loss + loss1
        loss.backward()
        optimizer.step()

        self.soft_update()

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
    
class HybridCritic(BaseModel):
    def __init__(self, encoder, embedding_size, device, kwargs):
        super(HybridCritic, self).__init__(encoder, embedding_size, device, kwargs)
        self.actor = nn.Sequential(
            nn.TransformerEncoderLayer(embedding_size, 2),
            nn.Linear(embedding_size, 2)
        )
        self.critic_encoder = copy.deepcopy(encoder)
        self.critic = nn.Sequential(
            nn.TransformerEncoderLayer(embedding_size, 2),
            nn.Linear(embedding_size, 1),
            Squeeze(-1),
            nn.AdaptiveAvgPool1d(1),
            Squeeze(-1)
        )

        self.target_critic_encoder = copy.deepcopy(self.critic_encoder)
        self.target_critic = copy.deepcopy(self.critic)

        self.networks.extend(["actor", "critic", "target_critic", "critic_encoder", "target_critic_encoder"])
        self.is_encoder.extend(["critic_encoder", "target_critic_encoder"])

    def forward(self, x):
        output = self.encoder(x, output_hidden_states=True)
        x = output["hidden_states"][-1]
        actor_logits = self.actor(x)
        return F.log_softmax(actor_logits, dim=2)
    
    def soft_update(self):
        self.target_critic_encoder.load_state_dict(utils.soft_update_dict(self.target_critic_encoder.state_dict(), self.critic_encoder.state_dict(), self.args.tau))
        self.target_critic.load_state_dict(utils.soft_update_dict(self.target_critic.state_dict(), self.critic.state_dict(), self.args.tau))

    def update(self,
               batch,
               optimizer,
               tokenizer,
               reward_generator,
               manager,
               ):
        for _ in range(self.args.explore_rounds):
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

            argmax_len = np.mean([len(word_tokenize(s)) for s in argmax_summaries])

            # Extend batch
            prob_batches, summary_batches, reward_batches, detail_batches, label_batches, ids_batches = sampling.k_samples(
                self.args, manager, tokenizer, reward_generator,
                input_ids, batch, probs,
                k_samples=self.args.explore_num,
                return_all=True
            )
            target_critics_batch = [self.target_critic(self.target_critic_encoder(ids, output_hidden_states=True)["hidden_states"][-1]).flatten().detach() for ids in ids_batches]

            # Update critic
            critic_input = self.critic_encoder(input_ids, output_hidden_states=True)["hidden_states"][-1]
            current_critics = self.critic(critic_input)
            target_critics = torch.stack(target_critics_batch).mean(dim=0)
            rewards = torch.tensor(reward_batches, dtype=torch.float).to(self.device)
            rewards = torch.mean(rewards, dim=0)
            target_critics = target_critics * self.args.gamma + rewards
            critic_loss = F.mse_loss(current_critics, target_critics)
            loss = critic_loss

            # Update actor
            current_critics = current_critics.detach()
            actor_loss = 0.
            entropy = 0.
            for sample_probs_i, rewards_i, target_critic_i in zip(prob_batches, reward_batches, target_critics_batch):
                rewards_i = torch.tensor(rewards_i, dtype=torch.float).to(self.device).detach()
                target_critic_i = target_critic_i.detach()
                advantage = rewards_i + self.args.gamma * target_critic_i - current_critics
                sample_probs_i = sample_probs_i.mean(dim=1)
                actor_loss_i = - (sample_probs_i * advantage).mean()
                entropy += - sample_probs_i.mean()
                actor_loss += actor_loss_i
            actor_loss = actor_loss / len(target_critics_batch)
            entropy = entropy / len(target_critics_batch)
            loss += actor_loss - self.args.entropy_weight * entropy
            if self.args.sample_aggregation == "max":
                loss1 = (a_reward - s_reward) * sample_probs.sum(1).mean()
            else:
                loss1 = 0.
                for sample_probs_i, s_rewards_i, a_rewards_i in zip(sample_data["probs"], sample_data["rewards"], sample_rewards):
                    s_reward_i = np.mean(s_rewards_i)
                    a_reward_i = np.mean(a_rewards_i)
                    loss_i = (a_reward_i - s_reward_i) * sample_probs_i.sum(1).mean()
                    loss1 += loss_i
                loss1 /= len(sample_data["rewards"])
            loss += loss1
            loss.backward()
            optimizer.step()

            self.soft_update()

            metrics = {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "entropy": entropy.mean().item()
            }
            manager.update_metrics(metrics)


            self.manager_update(manager,
                                loss,
                                a_reward,
                                s_reward,
                                sample_probs,
                                probs,
                                argmax_len,
                                argmax_details
                                )
            
            batch["document"] = argmax_summaries
            batch["input_ids"] = tokenizer(argmax_summaries)["input_ids"]
            # print(f"Round: {_}, Batch: {batch}")

        return probs, argmax_summaries, sample_summaries, argmax_details    

    def predict(self, texts, tokenizer):
        for i in range(self.args.exploit_num):
            input_ids = tokenizer(texts)["input_ids"]
            input_ids = pad_sequence(
                [torch.tensor(ids) for ids in input_ids], batch_first=True
            ).to(self.device)
            logits = self.forward(input_ids)
            argmax_labels = torch.argmax(logits, dim=2)
            texts = utils.labels_to_summary(input_ids, argmax_labels, tokenizer)
        return texts
