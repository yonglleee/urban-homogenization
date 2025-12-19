import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from datasets import BSVHistoryDataset,BSVHistoryMaskDataset
import models_mae
from util.datasets import build_transform

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare_model(chkpt_dir, arch='mae_vit_small_patch16'):
    """Prepare model and load checkpoint"""
    model = models_mae.__dict__[arch](norm_pix_loss=True)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def main(rank, world_size, args):
    """Main function with distributed support"""
    setup(rank, world_size)
    
    # 设置预处理转换
    args.input_size = 224
    transform_test = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载模型并包装为 DistributedDataParallel
    torch.cuda.set_device(rank)
    model = prepare_model(args.model_path).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()

    # 创建分布式sampler
    dataset = BSVHistoryDataset(args.data_path, transform=transform_test)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=32,
        pin_memory=True
    )

    features_data = []
    
    # 只在主进程显示进度条
    if rank == 0:
        dataloader = tqdm(dataloader)
    
    for batch in dataloader:
        imgs, files = batch
        imgs = imgs.to(rank, non_blocking=True)
        
        with torch.no_grad():
            loss, y, mask, latent = model(imgs, train=False) 
            loss = loss.cpu().numpy()
            cls_tokens = latent[:,:1,:].cpu().numpy()
        
        for file, l, feature in zip(files, loss, cls_tokens):
            feature_flat = feature.flatten().tolist()
            features_data.append({
                'PanoID': file,
                'loss': l,
                **{f'feature_{i}': feature_flat[i] for i in range(len(feature_flat))}
            })
    
    # 收集所有进程的结果
    all_features = [None] * world_size
    dist.all_gather_object(all_features, features_data)
    
    # 只在主进程保存结果
    if rank == 0:
        # 合并所有结果
        combined_features = []
        for features in all_features:
            combined_features.extend(features)
        
        df = pd.DataFrame(combined_features)
        df.to_csv(args.output_dir, index=False)
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using MAE")
    parser.add_argument("--data_path", type=str, default="/home/liyong/code/CityHomogeneity/data/baidu/V3/merged_test.csv", 
                       help="Root directory containing city images")
    parser.add_argument("--output_dir", type=str, default="/home/liyong/code/CityHomogeneity/data/baidu/V3/test_loss_feature.csv", 
                       help="Output directory to save feature CSV files")
    parser.add_argument("--model_path", type=str, 
                       default="/home/liyong/code/svpretrain/output_dir/checkpoint/mae_vit_small_bsv1m_v3/checkpoint-200.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction")
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID to use")
    args = parser.parse_args()

    # 获取可用GPU数量
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    # 使用torch.multiprocessing启动多进程
    torch.multiprocessing.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )