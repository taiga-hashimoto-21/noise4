"""
ホワイトノイズ（ガウシアンノイズ）の実装
測定器の熱雑音や電子回路の熱雑音を模擬
"""

import torch


def add_gaussian_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30):
    """
    パターン1: ガウシアンノイズ（ホワイトノイズ）
    測定器の熱雑音や電子回路の熱雑音を模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # ガウシアンノイズを生成（すべての周波数で一様）
    noise = torch.randn(points_per_interval) * interval_mean * noise_level
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

