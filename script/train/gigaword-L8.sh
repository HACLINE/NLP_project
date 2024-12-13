export CUDA_VISIBLE_DEVICES=6
python bin/train.py \
    --verbose \
    --config config/gigaword-L8.json \
    --device cuda