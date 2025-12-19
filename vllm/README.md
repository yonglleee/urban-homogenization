
# Homogenity Classification with Qwen2.5-VL & vLLM

This directory provides scripts for batch classification of urban streetscape images using the Qwen2.5-VL-7B-Instruct model, served via an OpenAI-compatible API using [vLLM](https://github.com/vllm-project/vllm).

## Prerequisites

- You must deploy the Qwen2.5-VL-7B-Instruct model using vLLM. See the [vLLM project](https://github.com/vllm-project/vllm) for installation and deployment instructions.
- The API service should be running and accessible (default: 127.0.0.1:8080). Example launch script: `qwen25vl_server.sh`.

## Main Files

- `homogenity_cls.py`: Main script for reading a CSV of image paths, sending images to the Qwen2.5-VL API, and saving classification results.
- `run_homogenity.sh`: Example shell script to run the classification pipeline.
- `qwen25vl_server.sh`: Example script to launch the vLLM server with Qwen2.5-VL.

## Usage

### 1. Prepare Input CSV
- The CSV must contain at least two columns: `panoid` (unique id) and `path` (image path, absolute or relative).

### 2. Start the vLLM Service
- Deploy the Qwen2.5-VL-7B-Instruct model with vLLM. Example:

```bash
bash qwen25vl_server.sh
```

### 3. Run the Classification Script

Example (see `run_homogenity.sh`):

```bash
bash run_homogenity.sh
```

Or run manually:

```bash
python3 homogenity_cls.py \
    /path/to/input.csv \
    /path/to/output.jsonl \
    127.0.0.1 \
    8080 \
    --base_path /your/image/root/
```

- `input.csv`: CSV file with `panoid` and `path` columns
- `output.jsonl`: Output file (one JSON per line: {"panoid": ..., "category": ...})
- `--base_path`: (Optional) Prefix for relative image paths

### 4. Output
- Results are saved as JSONL, one line per image, with fields `panoid` and `category`.
- Failures are saved to a `.failures.jsonl` file.

## Requirements
- Python 3.8+
- `fire`, `tqdm`, `pandas`, `openai`, `Pillow`, `cv2`, `requests`

Install dependencies:
```bash
pip install fire tqdm pandas openai pillow opencv-python requests
```

## Notes
- The script supports parallel processing for speed.
- Prompts are designed for urban scene classification with a fixed label set.
- See the top of `homogenity_cls.py` for prompt details and label definitions.
