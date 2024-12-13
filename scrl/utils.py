import numpy as np
import shutil
import json
import gzip
import random
import torch
from torch.nn.utils.rnn import pad_sequence


class TransformersTokenizerWrapper:
    def __init__(self, tokenizer):
        self.T = tokenizer

    def __call__(self, texts):
        token_ids_batch = self.T(texts)["input_ids"]
        tokens_batch = [[self.T._convert_id_to_token(id) for id in ids] for ids in token_ids_batch]
        tokens_batch = [[self.T.convert_tokens_to_string(t).strip() for t in tokens[1:-1]] for tokens in tokens_batch]
        return tokens_batch



def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def ask_rmdir(dir):
    val = input(
        f"WARNING: Proceed with deleting this directory: {dir} ? (yes|no) "
    )
    if val == "yes":
        shutil.rmtree(dir)


def load_numpy(path):
    with open(path, "rb") as f:
        x = np.load(f)
    return x


def save_numpy(x, path):
    with open(path, "wb") as f:
        np.save(f, x)


def batchify(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def move_generator(items, idx):
    if idx == 0:
        return
    else:
        for i, x in enumerate(items):
            if i >= idx - 1:
                break


def read_json(path):
    with open(path) as f:
        obj = json.load(f)
    return obj


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def write_jsonl(items, path, mode):
    with open(path, mode) as f:
        lines = [json.dumps(x) for x in items]
        f.write("\n".join(lines) + "\n")


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def read_jsonl_gz(path):
    with gzip.open(path) as f:
        for l in f:
            yield json.loads(l)

def labels_to_summary(input_batch, label_batch, tokenizer):
    summaries = []
    for input_ids, labels in zip(input_batch, label_batch):
        selected = [int(input_ids[i]) for i in range(len(input_ids))
                           if labels[i] == 1]
        summary = tokenizer.decode(selected, skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def labels_to_ids(input_batch, label_batch):
    idsl = []
    for input_ids, labels in zip(input_batch, label_batch):
        selected = [int(input_ids[i]) for i in range(len(input_ids))
                           if labels[i] == 1]
        idsl.append(selected)
    return pad_sequence(
        [torch.tensor(ids) for ids in idsl],
        batch_first=True
    ).to(input_batch[0].device)

def soft_update_dict(target, source, tau):
    for k, v in source.items():
        target[k] = tau * v + (1 - tau) * target[k]
    return target
    