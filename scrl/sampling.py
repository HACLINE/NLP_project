import torch
import random
import numpy as np
from collections import defaultdict
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from scrl.utils import labels_to_summary, labels_to_ids
from nltk import word_tokenize
from pprint import pprint


def sample_from_policy(
        input_ids,
        probs,
        device="cuda",
        force_diff=True,
        diff_trials=1000,
    ):
    """
    Sample labels from the distribution of the policy.

    args:
    - input_ids: torch.Tensor, shape (batch_size, seq_len)
    - probs: torch.Tensor, shape (batch_size, seq_len, vocab_size)
    - device: str, device
    - force_diff: bool, whether to force the sampled labels to be different from the argmax labels
    - diff_trials: int, number of trials to force the sampled labels to be different from the argmax labels

    return:
    - sample_probs: torch.Tensor, shape (batch_size, seq_len)
    - sample_labels: torch.Tensor, shape (batch_size, seq_len)
    """
    m = Categorical(probs)
    argmax_labels = torch.argmax(probs, dim=2)
    sample_labels = m.sample()

    if force_diff:
        for _ in range(diff_trials):
            if (argmax_labels == sample_labels).all():
                sample_labels = m.sample()
            else:
                break

    sample_probs = m.log_prob(sample_labels)
    return sample_probs, sample_labels


def best_of_k_samples(
        args,
        manager,
        tokenizer,
        reward_generator,
        input_ids,
        batch,
        probs,
        k_samples=50,
        return_all=False
    ):
    """
    Sample k samples from the policy and return the best one.

    args:
    - args: arguments containing device
    - manager: ?
    - tokenizer: tokenizer
    - reward_generator: reward generator
    - input_ids: torch.Tensor, shape (batch_size, seq_len)
    - batch: dict, batch data
    - probs: torch.Tensor, shape (batch_size, seq_len, vocab_size)
    - k_samples: int, number of samples to draw
    - return_all: bool, whether to return all samples
    """
    batch_size = probs.size(0)

    prob_batches = []
    summary_batches = []
    reward_batches = []
    detail_batches = []
    label_batches = []
    for _ in range(k_samples):
        sample_probs, sample_labels = sample_from_policy(
            input_ids,
            probs,
            device=args.device
        )
        sample_summaries = labels_to_summary(
            input_ids, sample_labels, tokenizer
        )
        sample_rewards, sample_details = reward_generator(
            batch["document"], sample_summaries
        )

        prob_batches.append(sample_probs)
        summary_batches.append(sample_summaries)
        reward_batches.append(sample_rewards)
        detail_batches.append(sample_details)
        label_batches.append(sample_labels)


    best_indices = []
    for i in range(batch_size):
        rewards = [reward_batches[j][i] for j in range(k_samples)]
        scored = sorted(enumerate(rewards), key=lambda x: x[1], reverse=True)
        best_idx = scored[0][0]
        best_indices.append(best_idx)

    sample_probs = torch.stack([prob_batches[j][i] for i, j in enumerate(best_indices)])
    sample_summaries = [summary_batches[j][i] for i, j in enumerate(best_indices)]
    sample_rewards = [reward_batches[j][i] for i, j in enumerate(best_indices)]
    sample_labels = torch.stack([label_batches[j][i] for i, j in enumerate(best_indices)])

    sample_details = []
    for i, j in enumerate(best_indices):
        detail_keys = sorted(detail_batches[0].keys())
        details = defaultdict(list)
        for k in detail_keys:
            details[k].append(detail_batches[j][k][i])
        sample_details.append(details)

    sample_data = {
        "probs": prob_batches,
        "rewards": reward_batches,
        "summaries": summary_batches,
        "details": detail_batches,
        "labels": label_batches,
    }
    return sample_probs, sample_summaries, sample_rewards, sample_details, sample_labels, sample_data

def k_samples(
        args,
        manager,
        tokenizer,
        reward_generator,
        input_ids,
        batch,
        probs,
        k_samples=50,
        return_all=False
    ):
    """
    Sample k samples from the policy.
    [[a1, b1, c1], [a2, b2, c2], ..., [ak, bk, ck]]

    args:
    - args: arguments containing device
    - manager: ?
    - tokenizer: tokenizer
    - reward_generator: reward generator
    - input_ids: torch.Tensor, shape (batch_size, seq_len)
    - batch: dict, batch data
    - probs: torch.Tensor, shape (batch_size, seq_len, vocab_size)
    - k_samples: int, number of samples to draw
    - return_all: bool, whether to return all samples
    """
    batch_size = probs.size(0)

    prob_batches = []
    summary_batches = []
    reward_batches = []
    detail_batches = []
    label_batches = []
    ids_batches = []
    for _ in range(k_samples):
        sample_probs, sample_labels = sample_from_policy(
            input_ids,
            probs,
            device=args.device
        )
        sample_summaries = labels_to_summary(
            input_ids, sample_labels, tokenizer
        )
        sample_ids = labels_to_ids(input_ids, sample_labels)
        sample_rewards, sample_details = reward_generator(
            batch["document"], sample_summaries
        )

        prob_batches.append(sample_probs)
        summary_batches.append(sample_summaries)
        reward_batches.append(sample_rewards)
        detail_batches.append(sample_details)
        label_batches.append(sample_labels)
        ids_batches.append(sample_ids)

    return prob_batches, summary_batches, reward_batches, detail_batches, label_batches, ids_batches
