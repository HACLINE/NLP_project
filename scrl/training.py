import argparse
import shutil
import logging
import random
import time
from pprint import pprint
from collections import defaultdict
from pathlib import Path

from scrl.rewards import load_rewards
from scrl.data import load_data_for_training
from scrl.config import load_config
from scrl.model import setup_model
from scrl.manager import TrainingManager
import scrl.utils as utils
import scrl.sampling as sampling

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from sklearn import preprocessing
import wandb
from torch.utils.tensorboard import SummaryWriter

from nltk import word_tokenize


def print_if(x, do_print=True):
    if do_print:
        print(x)

def check_gradient(model):
    is_zero = []
    is_none = []
    for name, param in list(model.named_parameters()):
        if (param.requires_grad):
            grad = param.grad
            if grad is None:
                is_none.append(name)
            else:
                gradsum = param.grad.sum().item()
                if gradsum == 0:
                    is_zero.append(name)
    print("zero-grad:", len(is_zero), is_zero)
    print("none-grad:", len(is_none), is_none)
    print()

def print_training_progress(args, manager, model, probs, argmax_summaries, sample_summaries, batch, argmax_details):
    print(f"[step: {manager.step}] [duration(s): {round(manager.total_seconds)}]")
    print(f"[example/s: {(args.batch_size * (manager.step + 1)) / manager.total_seconds:.3f}]")
    print(f"[s/step: {manager.total_seconds / (manager.step+1):.3f}]")
    print(f"[avg-loss: {manager.mean_metric('loss')}]")
    print(f"[avg-max-prob: {manager.mean_metric('mean_max_prob'):.3f}]")
    print(f"[avg-a-reward: {manager.mean_metric('argmax_reward'):.3f}]")
    print(f"[avg-s-reward: {manager.mean_metric('sample_reward'):.3f}]")
    print(f"[avg-len: {manager.mean_metric('argmax_len'):.1f}]")
    print()
    print(f"[a-reward: {manager.series['argmax_reward'][-1]:.3f}]")
    print(f"[s-reward: {manager.series['sample_reward'][-1]:.3f}]")
    print(f"[max-prob: {manager.series['mean_max_prob'][-1]:.3f}]")
    print()
    print("[sentences]")
    print("\n".join(batch["document"]))
    print("\n[current policy summaries]")
    print("\n".join(argmax_summaries))
    print("\n[sampled summaries]")
    print("\n".join(sample_summaries))
    print()
    print("Reward Breakdown:")
    pprint(argmax_details)
    print()
    check_gradient(model)
    print("="*100)
    if manager.wandb_run_id:
        wandb.log({
            "sentences": "\n".join(batch["document"]),
            "current_policy_summaries": "\n".join(argmax_summaries),
            "sampled_summaries": "\n".join(sample_summaries)
        }, step=manager.step)


def setup_dataset_indices(args, step):
    """
    Load pre-built indices that determine in which order we traverse a dataset.
    If we continue interrupted training state, we move indices accordingly.
    """
    dataset_indices = utils.batchify(
        utils.load_numpy(args.indices),
        args.batch_size
    )
    if step > 0:
        utils.move_generator(dataset_indices, step)
    return dataset_indices


def train(
        args,
        manager,
        model,
        tokenizer,
        reward_generator,
        dataset,
        dataset_indices,
        eval_func
    ):

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    n_train = len(dataset["train"])
    device = args.device
    model.train()
    manager.start_clock()

    for indices in dataset_indices:

        step = manager.step
        manager.total_seconds = time.time() - manager.start_time
        if args.max_train_steps and step >= args.max_train_steps + 1:
            break
        if args.max_train_seconds and manager.total_seconds >= args.max_train_seconds:
            break

        batch = dataset["train"][indices]

        probs, argmax_summaries, sample_summaries, argmax_details = model.update(
            batch,
            optimizer,
            tokenizer,
            reward_generator,
            manager
        )

        if args.eval_every != None and (step > 0 and step % args.eval_every == 0):
            eval_func(
                args, manager, model, tokenizer, reward_generator,
                dataset["validation"]
            )
            model.train()

        if args.save_every != None and (step % args.save_every == 0):
            manager.save_latest_model(model, step)
            manager.save_data()

        if args.print_every != None and (args.verbose and step % args.print_every == 0):
            print_training_progress(
                args, manager, model, probs,
                argmax_summaries, sample_summaries, batch,
                argmax_details
            )
        manager.step += 1


def setup_and_train(args, eval_func):

    print_if("loading model", args.verbose)
    manager, model = setup_model(args)

    print_if("loading tokenizer", args.verbose)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_id, device=args.device)

    print_if("loading rewards", args.verbose)
    reward_generator = load_rewards(args)
    print_if("rewards:", reward_generator.reward_names)

    print_if("loading dataset", args.verbose)
    dataset = load_data_for_training(tokenizer, args.loader, args.dataset)

    dataset_indices = setup_dataset_indices(args, manager.step)

    train(
        args,
        manager,
        model,
        tokenizer,
        reward_generator,
        dataset,
        dataset_indices,
        eval_func
    )
