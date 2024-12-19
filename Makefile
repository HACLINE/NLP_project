CONFIG ?= config/gigaword-ActorCritic.json
DEVICE ?= cuda
MODELDIR ?= data/models/all
# MODELDIR ?= data/models/example
# MODELDIR ?= data/models/allActorCritichalf
# MODELDIR ?= data/models/allActorCriticnew
# MODELDIR ?= data/models/half-compress
# MODELDIR ?= data/models/HybridCritictry
MODELDIR ?= data/models/newsroom-L11
# MODELDIR ?= data/models/gigaword-l11
# MODELDIR ?= data/models/allhalf
# MODELDIR ?= data/models/half2
TESTSET ?= data/test-data/broadcast.jsonl
HC_OUTPUT ?= data/hc-outputs/hc.L11.google.jsonl
PRED_ROUND ?= 1

# TRAINING

.PHONY: train
train:
	python bin/train.py --config config/example.json --device $(DEVICE) 
half-compress:
	python bin/train.py --config config/half-compress.json --device $(DEVICE)
half-compress2:
	python bin/train.py --config config/half-compress2.json --device $(DEVICE) 
half-compress3:
	python bin/train.py --config config/half-compress3.json --device $(DEVICE)
half-compress4:
	python bin/train.py --config config/half-compress4.json --device $(DEVICE)
half-compress5:
	python bin/train.py --config config/half-compress5.json --device $(DEVICE)
half-compress6:
	python bin/train.py --config config/half-compress6.json --device $(DEVICE)
train-hybrid-critic:
	python bin/train.py --config config/gigaword-HybridCritictry.json --device $(DEVICE)
train-actor-critic:
	python bin/train.py --config config/gigaword-ActorCritic.json --device $(DEVICE)
train-transformer-Q:
	python bin/train.py --config config/gigaword-TransformerQ.json --device $(DEVICE)
try:
	python bin/train.py --config config/gigaword-HybridCritictry.json --device $(DEVICE) 
all:
	python bin/train.py --config config/gigaword-all.json --device $(DEVICE)
half1:
	python bin/train.py --config config/half1.json --device $(DEVICE)
half2:
	python bin/train.py --config config/half2.json --device $(DEVICE)
old:
	python bin/train.py --config config/gigaword-actorcriticallold.json --device $(DEVICE)
newsroom-l11:
	python bin/train.py --config config/newsroom-l11.json --device $(DEVICE)
newsroom-L11:
	python bin/train.py --config config/newsroom-L11.json --device $(DEVICE)
gigaword-l11:
	python bin/train.py --config config/gigaword-l11.json --device $(DEVICE)
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

eval-all: eval-google eval-duc2004 eval-gigaword eval-broadcast eval-bnc
# EVALUATE HILL CLIMBING SEARCH

.PHONY: hc-eval-google
hc-eval-google:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/google.jsonl \
    	--outputs $(HC_OUTPUT)
eval-all-google:
	python bin/evaluate.py \
		--model-dir data/models/allActorCritic \
		--device $(DEVICE) \
		--pred-round $(PRED_ROUND) \
		--dataset data/test-data/google.jsonl > allactorcritic.txt
	python bin/evaluate.py \
		--model-dir data/models/allActorCriticnew \
		--device $(DEVICE) \
		--pred-round $(PRED_ROUND) \
		--dataset data/test-data/google.jsonl > allactorcriticnew.txt
	python bin/evaluate.py \
		--model-dir data/models/half-compress \
		--device $(DEVICE) \
		--pred-round $(PRED_ROUND) \
		--dataset data/test-data/google.jsonl > half-compress.txt
	python bin/evaluate.py \
		--model-dir data/models/all \
		--device $(DEVICE) \
		--pred-round $(PRED_ROUND) \
		--dataset data/test-data/google.jsonl > all.txt
	python bin/evaluate.py \
		--model-dir data/models/example \
		--device $(DEVICE) \
		--pred-round $(PRED_ROUND) \
		--dataset data/test-data/google.jsonl > example.txt
	python bin/evaluate.py \
		--model-dir data/models/half-compress2 \
		--device $(DEVICE) \
		--pred-round $(PRED_ROUND) \
		--dataset data/test-data/google.jsonl > half-compress2.txt


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
