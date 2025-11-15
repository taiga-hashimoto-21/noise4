"""
スパイクノイズ（バーストノイズ）の実装
特定の周波数帯域に集中的に発生するノイズ
電磁干渉や外乱を模擬
"""

import torch


def add_burst_noise(psd_data, interval_idx, noise_level=0.15, num_intervals=30, burst_ratio=0.3):
    """
    パターン3: バーストノイズ（スパイクノイズ）
    特定の周波数帯域に集中的に発生するノイズ
    電磁干渉や外乱を模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        burst_ratio: バーストが発生する割合（0.3 = 30%のポイントに大きなノイズ）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # バーストノイズを生成
    # ランダムに選んだポイントに大きなノイズを加える
    num_burst_points = int(points_per_interval * burst_ratio)
    burst_indices = torch.randperm(points_per_interval)[:num_burst_points]
    
    noise = torch.zeros(points_per_interval)
    # バーストポイントには大きなノイズ
    noise[burst_indices] = torch.randn(num_burst_points) * interval_mean * noise_level * 3.0
    # その他のポイントには小さなノイズ
    other_indices = torch.ones(points_per_interval, dtype=torch.bool)
    other_indices[burst_indices] = False
    noise[other_indices] = torch.randn(points_per_interval - num_burst_points) * interval_mean * noise_level * 0.3
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

