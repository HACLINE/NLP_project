export CUDA_VISIBLE_DEVICES=2
python bin/train.py \
    --verbose \
    --config config/gigaword-L8.json \
    --device cuda