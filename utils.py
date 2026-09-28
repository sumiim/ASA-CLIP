import random
import numpy as np
import torch
import logging
import os
from datetime import datetime

def set_seed(seed):
    """
    设置随机种子以确保实验的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(output_dir, log_name):
    """
    配置并返回一个日志记录器。
    
    Args:
        output_dir (str): 【修正】现在这个参数是 *完整* 的运行目录路径 (例如 ./outputs/ClipFit_...)
        log_name (str): 日志文件的名称 (例如 training_run.log)
    """
    # --- START: 修正 ---
    # 我们不再在这里创建带时间戳的子目录，
    # 因为 main.py 已经创建好了并作为 output_dir 传递了过来。
    # 我们直接使用这个传入的目录。
    exp_dir = output_dir 
    # --- END: 修正 ---

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 handler
    if not logger.handlers:
        # --- START: 修正 ---
        # 确保日志文件保存在正确的 exp_dir 中
        file_handler = logging.FileHandler(os.path.join(exp_dir, log_name))
        # --- END: 修正 ---
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式化器并添加到 handler
        # (修正：移除 INFO:utils: 这种多余的前缀，让日志更干净)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # --- END: 修正 ---
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加 handler 到 logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger


class CompactRunLog:
    """Short, agent-readable run log. Complements training_run.log (verbose)."""

    def __init__(self, run_dir, filename="run_summary.log"):
        self.path = os.path.join(run_dir, filename)
        self._file = open(self.path, "w", encoding="utf-8")
        self.write("# ASA-CLIP run_summary.log (agent-readable; no tqdm)")
        self.write(f"started={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write("status=running")

    def write(self, line=""):
        self._file.write(line.rstrip() + "\n")
        self._file.flush()

    def kv(self, **kwargs):
        parts = []
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, float):
                if 0.0 < abs(value) < 1e-3:
                    parts.append(f"{key}={value:.6g}")
                else:
                    parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        if parts:
            self.write(" ".join(parts))

    def metrics(self, prefix, metrics):
        if not metrics:
            return
        parts = [f"{prefix}_{k}={float(v):.4f}" for k, v in metrics.items()]
        self.write(" ".join(parts))

    def done(self, status="done"):
        if self._file.closed:
            return
        self.write(f"finished={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.write(f"status={status}")
        self._file.close()
