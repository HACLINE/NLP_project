CONFIG ?= config/gigaword-ActorCritic.json
DEVICE ?= cuda
# MODELDIR ?= data/models/all
# MODELDIR ?= data/models/example
MODELDIR ?= data/models/allActorCritic
# MODELDIR ?= data/models/allActorCriticnew
# MODELDIR ?= data/models/half-compress
# MODELDIR ?= data/models/HybridCritictry
TESTSET ?= data/test-data/broadcast.jsonl
HC_OUTPUT ?= data/hc-outputs/hc.L11.google.jsonl
PRED_ROUND ?= 1

# TRAINING

.PHONY: train
train:
	python bin/train.py --config $(CONFIG) --device $(DEVICE) 
half-compress:
	python bin/train.py --config config/half-compress.json --device $(DEVICE)
train-my:
	python bin/train.py --config config/gigaword-LinearQ.json --device $(DEVICE) 
# EVALUATING SCRL MODELS (predict + evaluate)
train-hybrid:
	python bin/train.py --config config/gigaword-HybridQ.json --device $(DEVICE)
train-hybrid-critic:
	python bin/train.py --config config/gigaword-HybridCritic.json --device $(DEVICE)
train-actor-critic:
	python bin/train.py --config config/gigaword-ActorCritic.json --device $(DEVICE)
train-transformer-Q:
	python bin/train.py --config config/gigaword-TransformerQ.json --device $(DEVICE)
try:
	python bin/train.py --config config/gigaword-HybridCritictry.json --device $(DEVICE) 
all:
	python bin/train.py --config config/gigaword-all.json --device $(DEVICE) 
old:
	python bin/train.py --config config/gigaword-actorcriticallold.json --device $(DEVICE)
train-actor-critic-all:
	python bin/train.py --config config/gigaword-actorcriticall.json --device $(DEVICE)
.PHONY: eval-google
eval-google:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/google.jsonl \
		--pred-round $(PRED_ROUND)


.PHONY: eval-duc2004
eval-duc2004:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/duc2004.jsonl \
		--max-chars 75 \
		--pred-round $(PRED_ROUND)


.PHONY: eval-gigaword
eval-gigaword:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/gigaword.jsonl \
		--pretokenized \
		--pred-round $(PRED_ROUND)


.PHONY: eval-broadcast
eval-broadcast:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/broadcast.jsonl \
		--pretokenized \
		--pred-round $(PRED_ROUND)


.PHONY: eval-bnc
eval-bnc:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/bnc.jsonl \
		--pretokenized \
		--pred-round $(PRED_ROUND)


# EVALUATE HILL CLIMBING SEARCH

.PHONY: hc-eval-google
hc-eval-google:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/google.jsonl \
    	--outputs $(HC_OUTPUT)


.PHONY: hc-eval-duc2004
hc-eval-duc2004:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/duc2004.jsonl \
	    --outputs $(HC_OUTPUT)


.PHONY: hc-eval-gigaword
hc-eval-gigaword:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/gigaword.jsonl \
	    --outputs $(HC_OUTPUT)


.PHONY: hc-eval-broadcast
hc-eval-broadcast:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/broadcast.jsonl \
	    --outputs $(HC_OUTPUT)


.PHONY: hc-eval-bnc
hc-eval-bnc:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/bnc.jsonl \
	    --outputs $(HC_OUTPUT)
