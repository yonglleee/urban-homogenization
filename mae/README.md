
# MAE-based Training Code

This directory contains training code based on Masked Autoencoders (MAE).

## Main Files

- `main_pretrain.py`, `main_pretrain_bsv1.py`: MAE pre-training scripts
- `main_finetune.py`, `main_linprobe.py`: Fine-tuning and linear probe scripts
- `main_test_baidu.py`: Testing script for Baidu street view data
- `models_mae.py`, `models_vit.py`: Model definitions
- `datasets.py`: Dataset loading
- `util/`: Training utilities

## Usage Example

Pre-training:
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=1 main_pretrain_bsv1.py \
    --batch_size 512 \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.75 \
    --epochs 300 \
    --data_path <train_csv> \
    --output_dir <output_dir>
```

Testing:
```bash
python main_test_baidu.py \
    --data_path <test_csv> \
    --output_dir <output_csv> \
    --model_path <checkpoint.pth> \
    --batch_size 256 \
    --gpu_id 0
```

## Reference
- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://github.com/facebookresearch/mae)
