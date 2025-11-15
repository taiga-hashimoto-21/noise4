"""
ピンクノイズ（1/fノイズ）の実装
低周波数ほど大きくなるノイズ（フリッカーノイズ）
測定器のドリフトや環境変動を模擬
"""

import torch


def add_pink_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30):
    """
    パターン2: 1/fノイズ（ピンクノイズ）
    低周波数ほど大きくなるノイズ（フリッカーノイズ）
    測定器のドリフトや環境変動を模擬
    
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
    
    # 1/fノイズを生成
    # 周波数が低いほど（インデックスが小さいほど）ノイズが大きくなる
    frequencies = torch.arange(1, points_per_interval + 1, dtype=torch.float32)
    # 1/sqrt(f) の重みをかける（低周波ほど大きい）
    weights = 1.0 / torch.sqrt(frequencies)
    weights = weights / weights.max()  # 正規化
    
    # ガウシアンノイズに周波数依存の重みをかける
    base_noise = torch.randn(points_per_interval) * interval_mean * noise_level
    noise = base_noise * weights
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

