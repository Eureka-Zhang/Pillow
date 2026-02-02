# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
#===========================
#mel频谱 python3 MEL_preprocess.py --input_dir audio_wav/ --output_dir features_mel/ --preset Mel --num_workers 4
#paper python3 MEL_preprocess.py --input_dir audio_wav/ --output_dir features_mel/ --preset paper --num_workers 4
# python3 MEL_preprocess.py --input_dir snoring/ --output_dir dataset_mel/ --preset Mel --num_workers 4
#=============================
PRESET_CONFIGS = {
    "paper": {
        "sr": 32000,
        "duration": 3,
        "hop_length": 320,  # 10ms @32kHz
        "n_fft": 800,       # 25ms @32kHz
        "n_mels": 128,
        "window": "hamming",
        "preemphasis": False,
        "normalize": True,
        # SpecAugment 参数
        "freq_mask_param": 48,
        "time_mask_param": 48, 
        "num_freq_masks": 2,
        "num_time_masks": 2
    },
    "Mel": {
        "sr": 16000,
        "duration": 3.0,
        "hop_length": 160,  # 10ms @16kHz
        "n_fft": 400,       # 25ms @16kHz
        "n_mels": 80,
        "window": "hamming",
        "preemphasis": True,
        "normalize": True,
        # SpecAugment 参数
        "freq_mask_param": 48,
        "time_mask_param": 48, 
        "num_freq_masks": 2,
        "num_time_masks": 2
    }
}


class SpecAugment:
    """
    SpecAugment: 一种简单的音频数据增强方法
    参考: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(self, freq_mask_param=48, time_mask_param=48, num_freq_masks=2, num_time_masks=2):
        self.freq_mask_param = freq_mask_param  # 频率掩码最大长度
        self.time_mask_param = time_mask_param  # 时间掩码最大长度
        self.num_freq_masks = num_freq_masks    # 频率掩码数量
        self.num_time_masks = num_time_masks    # 时间掩码数量
    
    def __call__(self, spectrogram):
        """
        对频谱图应用SpecAugment
        
        Args:
            spectrogram: 输入频谱图 (freq_bins, time_frames)
        
        Returns:
            augmented_spectrogram: 增强后的频谱图
        """
        augmented = spectrogram.copy()
        
        # 应用频率掩码
        for _ in range(self.num_freq_masks):
            augmented = self.frequency_masking(augmented)
        
        # 应用时间掩码  
        for _ in range(self.num_time_masks):
            augmented = self.time_masking(augmented)
        
        return augmented
    
    def frequency_masking(self, spectrogram):
        """应用频率掩码"""
        freq_bins, time_frames = spectrogram.shape
        
        # 随机选择掩码长度和位置
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, freq_bins - f)
        
        # 应用掩码
        spectrogram[f0:f0+f, :] = 0
        return spectrogram
    
    def time_masking(self, spectrogram):
        """应用时间掩码"""
        freq_bins, time_frames = spectrogram.shape
        
        # 随机选择掩码长度和位置
        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, time_frames - t)
        
        # 应用掩码
        spectrogram[:, t0:t0+t] = 0
        return spectrogram

    def time_warping(self, spectrogram, W=5):
        """
        时间弯曲（可选）
        Args:
            W: 时间弯曲参数
        """
        freq_bins, time_frames = spectrogram.shape
        
        # 随机选择弯曲点
        point = random.randint(W, time_frames - W)
        # 随机选择弯曲距离
        dist = random.randint(-W, W)
        
        # 应用时间弯曲
        left = spectrogram[:, :point]
        center = spectrogram[:, point:point+abs(dist)]
        right = spectrogram[:, point+abs(dist):]
        
        if dist > 0:
            # 拉伸
            center = librosa.effects.time_stretch(center, rate=1.2)
            center = center[:, :dist]
        else:
            # 压缩
            center = librosa.effects.time_stretch(center, rate=0.8)
            center = center[:, :abs(dist)]
        
        # 重新组合
        warped = np.concatenate([left, center, right], axis=1)
        
        # 如果形状不匹配，进行调整
        if warped.shape[1] != time_frames:
            if warped.shape[1] > time_frames:
                warped = warped[:, :time_frames]
            else:
                warped = np.pad(warped, ((0, 0), (0, time_frames - warped.shape[1])), 
                               mode='constant')
        
        return warped
    

def preprocess_audio_mel(audio_path, config):
    """
    对音频进行预处理，输出Mel频谱图
    """
    try:
        y, sr = librosa.load(audio_path, sr=config["sr"])

        # 截取或补零
        if(config["duration"] != 999):
            target_len = int(sr * config["duration"])
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')

        # 高频预加重（可选）
        if config.get("preemphasis", False):
            y = librosa.effects.preemphasis(y)

        # 计算Mel谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=config["sr"],
            n_mels=config["n_mels"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            win_length=config.get("win_length", config["n_fft"]),
            window=config["window"],
            power=2.0
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        if config["augment"]:
            spec_augment = SpecAugment(
                freq_mask_param=config.get("freq_mask_param", 48),
                time_mask_param=config.get("time_mask_param", 48),
                num_freq_masks=config.get("num_freq_masks", 2),
                num_time_masks=config.get("num_time_masks", 2)
            )
            log_mel = spec_augment(log_mel)
        
        if config.get("normalize", True):
            # 标准化    
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

        return log_mel

    except Exception as e:
        print(f"processing {audio_path} fails: {e}")
        return None


def process_one_file_mel(args_tuple):
    """用于多进程的单文件处理函数"""
    fpath, input_dir, output_dir, params = args_tuple
    rel_path = os.path.relpath(fpath, input_dir)
    save_path = os.path.join(output_dir, rel_path.replace(".wav", ".npy"))

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mel_feat = preprocess_audio_mel(
        fpath,
        config=params  # 直接传递配置字典
    )

    if mel_feat is not None:
        np.save(save_path, mel_feat)
        return fpath
    return None


def batch_preprocess_mel(input_dir, output_dir, config, num_workers=4):
    """
    批量预处理音频为Mel谱图
    
    Args:
        input_dir: 输入音频目录
        output_dir: 输出特征目录
        config: 处理配置
        num_workers: 并行进程数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置文件
    config_save_path = os.path.join(output_dir, "preprocess_config.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 支持多种音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    
    # 递归收集所有音频文件
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in audio_extensions:
                audio_files.append(os.path.join(root, f))
    
    print(f"Found {len(audio_files)} audio files to process")
    print(f"Configuration: {config}")
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # 准备任务
    tasks = [(fpath, input_dir, output_dir, config) for fpath in audio_files]
    
    # 使用多进程加速
    completed_count = 0
    failed_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one_file_mel, task): task for task in tasks}
        
        with tqdm(total=len(futures), desc="Processing", ncols=80) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        completed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"Task failed: {e}")
                
                pbar.update(1)
                pbar.set_postfix(completed=completed_count, failed=failed_count)
    
    print(f"Processing completed!")
    print(f"Success: {completed_count}, Failed: {failed_count}")
    print(f"Features saved to: {output_dir}")
    print(f"Configuration saved to: {config_save_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="通用Mel频谱音频预处理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 输入输出参数
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入音频文件夹路径（可含多层子目录）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出特征保存路径")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="单个音频文件路径（可选）")
    
    # 预设模式
    parser.add_argument("--preset", type=str, choices=["paper", "Mel"], default="Mel",
                        help="使用预设配置模式")
    
    # 音频处理参数
    parser.add_argument("--sr", type=int, help="采样率 (Hz)")
    parser.add_argument("--duration", type=float, default=3, help="音频时长（秒）")
    parser.add_argument("--hop_length", type=int, help="STFT跳帧长度")
    parser.add_argument("--n_fft", type=int, help="FFT窗口长度")
    parser.add_argument("--n_mels", type=int, help="Mel频带数")
    parser.add_argument("--window", type=str, choices=["hamming", "hann", "blackman"],
                        help="窗函数类型")
    parser.add_argument("--preemphasis", action="store_true",default=False,
                        help="启用预加重")
    parser.add_argument("--no_normalize", action="store_true",default=True,
                        help="禁用标准化")
    #数据增强
    parser.add_argument("--augment", action="store_true", default=False,
                        help="启用数据增强（SpecAugment）")
    parser.add_argument("--freq_mask_param", type=int, default=48,
                        help="频率掩码参数")    
    parser.add_argument("--time_mask_param", type=int, default=48,
                        help="时间掩码参数")
    parser.add_argument("--num_freq_masks", type=int, default=2,
                        help="频率掩码数量")
    parser.add_argument("--num_time_masks", type=int, default=2,
                        help="时间掩码数量")
    
    # 系统参数
    parser.add_argument("--num_workers", type=int, default=4,
                        help="并行进程数")
    parser.add_argument("--save_format", type=str, choices=["npy", "npz"], default="npy",
                        help="特征保存格式")
    
    return parser.parse_args()


def get_config_from_args(args):
    """根据命令行参数获取配置"""
    if args.preset and args.preset in PRESET_CONFIGS:
        config = PRESET_CONFIGS[args.preset].copy()
    else:
        config = PRESET_CONFIGS["default"].copy()
    
    # 用命令行参数覆盖预设配置
    if args.sr is not None:
        config["sr"] = args.sr
    if args.duration is not None:
        config["duration"] = args.duration
    if args.hop_length is not None:
        config["hop_length"] = args.hop_length
    if args.n_fft is not None:
        config["n_fft"] = args.n_fft
    if args.n_mels is not None:
        config["n_mels"] = args.n_mels
    if args.window is not None:
        config["window"] = args.window
    if args.preemphasis is not None:
        config["preemphasis"] = args.preemphasis
    if args.no_normalize:
        config["normalize"] = False
    if args.augment is not None:
        config["augment"] = args.augment
    if args.freq_mask_param is not None:
        config["freq_mask_param"] = args.freq_mask_param
    if args.time_mask_param is not None:
        config["time_mask_param"] = args.time_mask_param
    if args.num_freq_masks is not None:
        config["num_freq_masks"] = args.num_freq_masks  
    if args.num_time_masks is not None:
        config["num_time_masks"] = args.num_time_masks  
    
    
    # 调试信息：打印命令行参数和最终配置
    print(f"命令行参数: {args}")
    print(f"最终配置: {config}")
    
    return config


if __name__ == "__main__":
    args = get_args()
    config = get_config_from_args(args)
    # 单文件模式
    if args.audio_path:
        print(f"process single file: {args.audio_path}")
        feat = preprocess_audio_mel(args.audio_path, config=config)
        if feat is not None:
            save_path = os.path.join(args.output_dir, os.path.basename(args.audio_path).replace(".wav", ".npy"))
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(save_path, feat)
            print(f"features saved to {save_path}")
    else:
        # 批量模式
        batch_preprocess_mel(args.input_dir, args.output_dir, config=config, num_workers=args.num_workers)
