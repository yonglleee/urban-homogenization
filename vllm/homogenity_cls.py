import sys

import io
import fire
import json
import os
import tqdm
import tarfile
import zipfile
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import cv2
import multiprocessing
from functools import partial
import time
from typing import List, Dict
import requests

SYSTEM_PROMPT = (
    "You are an urban environment classification specialist. Analyze each streetscape carefully, then report the most"
    " plausible scene category using precise, professional language. Focus on functional zoning cues and justify"
    " your decision succinctly."
)

USER_PROMPT = (
    "You will be given a streetscape image. Your task is to classify the image using exactly ONE of the following 6 main labels.\n\n"
    "If and ONLY IF the image does not clearly fit into any of the 6 main labels, you must create a new label using the format `other-xxx`,"
    " where `xxx` is a brief, specific, lowercase description of the scene (use underscores for spaces).\n\n"
    "---\n"
    "MAIN LABELS:\n"
    "1. HIGH_DENSITY_COMMERCIAL: Central Business District (CBD) scenes with dense skyscrapers, modern high-rises, and corporate offices.\n"
    "2. GENERAL_COMMERCIAL: Street-level retail areas with many storefronts, shops, and restaurants; less dense than a CBD.\n"
    "3. HIGH_DENSITY_RESIDENTIAL: Scenes dominated by large, multi-story apartment buildings and residential towers.\n"
    "4. LOW_DENSITY_RESIDENTIAL: Suburban streets, single-family homes, townhouses, or smaller residential buildings with more open space.\n"
    "5. INDUSTRIAL: Factories, warehouses, logistics centers, container yards, and other industrial facilities.\n"
    "6. PARK_OPEN_SPACE: Public parks, manicured green spaces, waterfronts, public plazas, and other open recreational areas.\n\n"
    "---\n"
    "OTHER LABEL EXAMPLES:\n"
    "- If the image shows a major highway interchange, use: other-highway\n"
    "- If the image is inside an airport terminal, use: other-airport_terminal\n"
    "- If the image shows a university campus, use: other-university_campus\n"
    "- If the image is a construction site, use: other-construction_site\n\n"
    "---\n"
    "OUTPUT INSTRUCTIONS:\n"
    "Output only the single chosen label — no explanations, no punctuation, no extra text, and no surrounding whitespace.\n\n"
    "For example, for an image of a quiet suburban street with houses, your output must be exactly:\n"
    "LOW_DENSITY_RESIDENTIAL"
)


def get_random_prompts():
    """Return the fixed system and user prompt for scene classification."""
    return SYSTEM_PROMPT, USER_PROMPT





class Qwen25VL:
  def __init__(self, service="Qwen2.5_VL_7B", ip=None ,port=8080) -> None:
    self.service = service
    self.ip = ip
    self.port = port
    self.client = None

  def __call__(self, content, ip=None):

    openai_api_key = "EMPTY"
    openai_api_base = f"http://{self.ip}:{self.port}/v1"
    client = OpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
    )
    return self.req_agent(client, content)

  def req_agent(self, client, content):
    """
    content:[
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": base64_qwen
                      },
                  },
                  {"type": "text", "text": prompt},
              ],
    """
    chat_response = client.chat.completions.create(
      model="Qwen2.5-VL-7B-Instruct",
      messages=[
        {
          "role": "user",
          "content": content,
        },
      ],
    )
    # print("Chat response:", chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content
    

def prepare_image_with_pil(img):
    # # 使用PIL打开图片
    img = Image.open(img)
    
    # 确保图片是RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 保存为PNG格式
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64


def call_api(processed_data: Dict, ip: str, port: int, retry_count: int = 1):
    """处理单个API调用，单图英文prompt"""
    img_path = processed_data["img_path"]
    user_prompt_template = processed_data["user_prompt_template"]
    system_prompt = processed_data["system_prompt"]
    
    # 先检查文件是否存在
    if not os.path.exists(img_path):
        print(f"图片文件不存在: {img_path}")
        return None
        
    for attempt in range(retry_count):
        try:
            img_base64 = prepare_image_with_pil(img_path)
            
            formatted_user_prompt = user_prompt_template
            
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": formatted_user_prompt
                },
                {
                    "type": "text",
                    "text": system_prompt.strip()
                }
            ]
            model = Qwen25VL(service="Qwen2.5-VL-7B-Instruct",ip=ip, port=port)
            response = model(content)
            # print(f"Got response for {img_path}: {response[:100]}...")
            
            try:
                # response_data = json.loads(response)
                response_data = response
                return response_data, processed_data
            except json.JSONDecodeError:
                print(f"JSON解析错误: {response[:100]}...")
                if attempt < retry_count - 1:
                    continue
                return None
        except Exception as e:
            print(f"API调用失败, 重试中 ({attempt + 1}/{retry_count}): {str(e)}")
            if attempt < retry_count - 1:
                time.sleep(2)
            else:
                print(f"达到最大重试次数: {img_path}")
                return None
    return None

def process_api_batch(batch_data: List[Dict], ip: str, port: int, result_queue=None):
    """处理一批API调用，返回成功和失败样本"""
    results = []
    failures = []
    pid = os.getpid()
    print(f"Batch processing started in PID {pid} with {len(batch_data)} items")
    
    try:
        for i, data in enumerate(tqdm.tqdm(batch_data, desc=f"API batch (PID: {pid})", disable=True)):
            try:
                result = call_api(data, ip, port)
                if result is not None:
                    results.append(result)
                    if result_queue is not None:
                        result_queue.put({'type': 'result', 'data': result})
                    
                    if (i+1) % 20 == 0:
                        print(f"PID {pid}: Processed {i+1}/{len(batch_data)}, success: {len(results)}, fail: {len(failures)}")
                else:
                    failures.append(data)
                    if result_queue is not None:
                        result_queue.put({'type': 'failure', 'data': data})
            except Exception as e:
                print(f"PID {pid}: Error processing item {i}: {str(e)}")
                failures.append(data)
                if result_queue is not None:
                    result_queue.put({'type': 'failure', 'data': data})
    except Exception as e:
        print(f"PID {pid}: Batch processing error: {str(e)}")
        
    print(f"PID {pid}: Batch complete. Results: {len(results)}, Failures: {len(failures)}")
    return results, failures


def writer_process(result_queue, save_path, fail_path, total_tasks, num_workers, already_processed=0):
    """统一写入结果和失败样本，支持进度条"""
    workers_finished = 0
    result_count = 0
    fail_count = 0
    print(f"Writer PID: {os.getpid()} started. Expecting {num_workers} workers to finish.")
    
    # 计算真实的总任务数（包括已处理的）
    real_total = total_tasks + already_processed
    
    with open(save_path, 'a+') as f_result, open(fail_path, 'a+') as f_fail, tqdm.tqdm(total=real_total, initial=already_processed, desc="Total Progress") as pbar:
        while workers_finished < num_workers:
            try:
                # 使用超时防止一直阻塞
                try:
                    item = result_queue.get(timeout=30)
                except:
                    print(f"Queue timeout. Workers finished: {workers_finished}/{num_workers}")
                    if workers_finished >= num_workers:
                        break
                    continue

                    
                if item is None:
                    workers_finished += 1
                    print(f"Writer: received poison pill. Finished workers: {workers_finished}/{num_workers}")
                    continue
                
                if item['type'] == 'result':
                    response_data, processed_data = item['data']
                    panoid = processed_data.get("panoid") or processed_data.get("img_path")
                    category_text = response_data["caption"] if isinstance(response_data, dict) and "caption" in response_data else str(response_data)
                    new_item = {
                        "panoid": panoid,
                        "category": category_text.strip()
                    }
                    f_result.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    f_result.flush()
                    result_count += 1
                    
                    if result_count % 20 == 0:
                        print(f"Writer: processed {result_count} results")
                
                elif item['type'] == 'failure':
                    f_fail.write(json.dumps(item['data']) + '\n')
                    f_fail.flush()
                    fail_count += 1
                    
                pbar.update(1)
                
            except Exception as e:
                print(f"Writer exception: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印详细堆栈
    
    print(f"Writer finished. Total results: {result_count}, failures: {fail_count}")


def worker_process(batch_data, result_queue, ip, port):
    """worker只负责处理API，把结果和失败样本放入队列"""
    worker_pid = os.getpid()
    print(f"Worker PID: {worker_pid} started with batch size: {len(batch_data)}")
    try:
        # 直接传递结果队列给process_api_batch，实现实时写入
        results, failures = process_api_batch(batch_data, ip, port, result_queue)
        print(f"Worker PID: {worker_pid} completed: {len(results)} results, {len(failures)} failures")
        # 结果已经在处理过程中写入队列，不需要在这里再次写入
    except Exception as e:
        print(f"Worker PID: {worker_pid} exception: {str(e)}")
    finally:
        result_queue.put(None)  # 发送毒丸信号


def process_api_calls_parallel(processed_data, save_path, ip, port, num_processes, already_processed=0):
    """并行处理API调用，采用producer-consumer模式"""
    print(f"Starting parallel processing with {num_processes} processes")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # 如果没有新数据要处理，直接返回
    if not processed_data:
        return None
    
    # 检查是否存在输出文件，如果存在则追加，否则创建新文件
    file_exists = os.path.exists(save_path)
    if not file_exists:
        # 如果文件不存在，创建空文件
        with open(save_path, 'w') as f:
            f.write("")
        print(f"Created new output file: {save_path}")
    else:
        print(f"Will append to existing file: {save_path}")
    
    # 计算每个进程的数据批次
    batch_size = max(1, len(processed_data) // num_processes)
    batches = [processed_data[i:i + batch_size] for i in range(0, len(processed_data), batch_size)]
    actual_num_processes = len(batches)  # 实际创建的进程数可能少于请求的数量
    
    print(f"Split {len(processed_data)} items into {len(batches)} batches (requested {num_processes} processes)")
    
    # 设置多进程共享队列
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Set start method warning: start method already set")
    
    from multiprocessing import Process, Manager
    manager = Manager()
    # 队列大小计算：进程数 × 缓冲倍数 (10-20)
    # 100进程 × 10 = 1000 (保守) 或 100进程 × 15 = 1500 (平衡)
    queue_size = min(num_processes * 15, 2000)  # 最大不超过2000
    result_queue = manager.Queue(maxsize=queue_size)
    fail_path = save_path + ".failures.jsonl"
    total_tasks = len(processed_data)
    num_workers = len(batches)
    
    # 启动writer进程，传递已处理的数量
    writer = Process(target=writer_process, args=(result_queue, save_path, fail_path, total_tasks, num_workers, already_processed))
    writer.start()
    print(f"Writer process started with PID: {writer.pid}")
    
    # 启动worker进程
    workers = []
    for i, batch in enumerate(batches):
        worker = Process(target=worker_process, args=(batch, result_queue, ip, port))
        workers.append(worker)
        worker.start()
        print(f"Worker {i+1}/{len(batches)} started with PID: {worker.pid}")
    
    # 等待所有worker完成
    for i, worker in enumerate(workers):
        worker.join()
    
    # 等待writer完成
    writer.join()
    
    print(f'All captions saved. Failures saved to {fail_path}')
    return None

def get_processed_images(save_path):
    """获取已经处理过的图片路径集合"""
    processed_images = set()
    if os.path.exists(save_path):
        print(f"Found existing output file: {save_path}")
        with open(save_path, 'r') as f:
            line_count = 0
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # 兼容旧结构与新结构
                    panoid = data.get('panoid')
                    if panoid:
                        processed_images.add(str(panoid))
                        line_count += 1

                    images = data.get('images')
                    if isinstance(images, list) and images:
                        processed_images.add(str(images[0]))
                        line_count += 1
                except:
                    continue
        print(f"Found {len(processed_images)} already processed images")
    return processed_images

def main(data_path, save_path, ip, port, base_path=None):
    import pandas as pd
    print("Stage1: Reading CSV and Building Prompts...")
    header_cols = pd.read_csv(data_path, nrows=0).columns
    required_cols = {"panoid", "path"}
    missing_cols = required_cols - set(header_cols)
    if missing_cols:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required_cols))}. Missing: {', '.join(sorted(missing_cols))}")

    df = pd.read_csv(data_path, usecols=list(required_cols))
    # 只取前100行测试
    # df = df.head(100)
    
    # 获取已处理的图片
    processed_images = get_processed_images(save_path)
    # 识别图像路径和唯一标识列
    if base_path is None:
        default_base_path = "/data_nas/liyong/"
        base_path = os.environ.get("HOMOGENITY_IMAGE_BASE", default_base_path)

    if base_path is not None:
        base_path = str(base_path).strip()
        if not base_path:
            base_path = None
    processed_data = []
    skipped_count = 0
    for idx, row in df.iterrows():
        if pd.isna(row["path"]):
            continue

        raw_img_path = str(row["path"]).strip()
        if not raw_img_path:
            continue

        if os.path.isabs(raw_img_path) or base_path is None:
            full_img_path = os.path.normpath(raw_img_path)
        else:
            full_img_path = os.path.normpath(os.path.join(base_path, raw_img_path))

        if pd.isna(row["panoid"]):
            continue

        panoid = str(row["panoid"]).strip()
        
        # 检查是否已经处理过
        if full_img_path in processed_images or (panoid and panoid in processed_images):
            skipped_count += 1
            continue
            
        # 为每个样本随机选择 prompts
        system_prompt, user_prompt_template = get_random_prompts()
        
        processed_data.append({
            "img_path": full_img_path,
            "panoid": panoid,
            "user_prompt_template": user_prompt_template,
            "system_prompt": system_prompt
        })
    
    print(f"Total images in CSV: {len(df)}")
    print(f"Already processed: {len(processed_images)}")
    print(f"Skipped (already done): {skipped_count}")
    print(f"New images to caption: {len(processed_data)}")
    
    if len(processed_data) == 0:
        print("No new images to process. All images have been captioned!")
        return
    
    final_results = process_api_calls_parallel(
        processed_data,
        save_path,
        ip=ip,
        port=port,
        num_processes=16,
        already_processed=len(processed_images)  # 传递已处理的数量
    )


if __name__ == "__main__":
    fire.Fire(main)