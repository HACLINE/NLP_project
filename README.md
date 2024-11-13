

# NLP_project

The code is from [rl-sentence-compression](https://github.com/complementizer/rl-sentence-compression), which is the official repo for the ACL 2022 paper [Efficient Unsupervised Sentence Compression by Fine-tuning Transformers with Reinforcement Learning](https://arxiv.org/abs/2205.08221).

See `setup.md` for setup instructions.

The following parts are subsets of the original README:

### Training a new model

A new model needs a new config file (examples in [config](config)) for various settings, e.g. training dataset, reward functions, model directory, steps.


`python bin/train.py --verbose --config config/example.json --device cuda`

You can also change the device to `cpu` to try it out locally.

Training can be interrupted with `Ctrl+C` and continued by re-running the same command which will pick up from the latest saved checkpoint. Add `--fresh` to delete the previous training progress and start from scratch.


### Evaluation

The evaluation results can be replicated with the following Make commands, which run with slightly different settings depending on the dataset:

```bash
make eval-google MODELDIR=data/models/newsroom-L11
make eval-duc2004 MODELDIR=data/models/newsroom-L11
make eval-gigaword MODELDIR=data/models/gigaword-L8
make eval-broadcast MODELDIR=data/models/newsroom-P75
make eval-bnc MODELDIR=data/models/newsroom-P75
```

To evaluate on a custom dataset, check out [bin/evaluate.py](bin/evaluate.py) and its arguments.


### Hill Climbing Baseline

They implemented a search-based baseline for sentence compression using hill climbing, based on [Discrete Optimization for Unsupervised Sentence Summarization with Word-Level Extraction](https://arxiv.org/abs/2005.01791).  A difference to the original method is that they only restart the search if no unknown neighbour state can be found, i.e. dynamically instead of in equal-paced intervals.

**Producing summaries**<br>
The budget of search steps is controlled with `--steps`.
```bash
python bin/run_hc.py \
    --config config/hc.json \
    --steps 10 \
    --target-len 11 \
    --dataset data/test-data/google.jsonl \
    --output data/hc-outputs/example.jsonl \
    --device cpu
```


**Evaluation** <br>

For datasets used in the paper:
```bash
make hc-eval-google HC_OUTPUT=data/hc-outputs/hc.L11.google.jsonl
make hc-eval-duc2004 HC_OUTPUT=data/hc-outputs/hc.L11.duc2004.jsonl
make hc-eval-gigaword HC_OUTPUT=data/hc-outputs/hc.L8.gigaword.jsonl
make hc-eval-broadcast HC_OUTPUT=data/hc-outputs/hc.P75.broadcast.jsonl
make hc-eval-bnc HC_OUTPUT=data/hc-outputs/hc.P75.bnc.jsonl
```

Example for custom dataset:
```
python bin/evaluate_hc_output.py \
    --dataset data/test-data/google.jsonl \
    --outputs data/hc-outputs/hc.L11.google.jsonl
```