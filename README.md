# Urban Homogenization Project

This repository provides code and tools for urban homogeneity analysis, including MAE-based training/inference, large-scale scene classification, and comprehensive data analysis and visualization.

## Data Download


The required data is shared via [Baidu Drive](https://pan.baidu.com/s/14V7qKHpZyzcjGAu9RSYz9w?pwd=4ekn) (Extraction code: `4ekn`).

Folder name: `UrbanHomogenization`

Please download and extract the data to the `data/` directory in this repository.

---

## Repository Structure

- `mae/`  
	MAE (Masked Autoencoder) training and inference code for urban image feature learning. Includes scripts for pretraining, fine-tuning, and feature extraction.

- `vllm/`  
	Scene classification using large vision-language models (Qwen2.5-VL) via vLLM. Includes scripts for batch classification of urban scenes and server deployment.

- `analysis/`  
	Data analysis and visualization. Contains notebooks and scripts for clustering, preprocessing, statistical analysis, and figure generation for publications.

- `data/`  
	Data directory (not included in the repo, see download instructions above). Contains all shapefiles, CSVs, and intermediate results required for experiments.

---

## Submodule Descriptions

- **mae/**: MAE-based model training, fine-tuning, and inference for urban imagery.
- **vllm/**: Large-scale scene classification using Qwen2.5-VL and vLLM server.
- **analysis/**: Data preprocessing, clustering, statistical analysis, and visualization.

---

## Quick Start

1. Download and extract the data as described above.
2. Install dependencies for each submodule (see their respective README files).
3. Follow the workflows in each submodule for training, inference, classification, and analysis.

---

## Contact

For questions or collaboration, please contact: yong.li@connect.ust.hk

