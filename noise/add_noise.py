"""
測定系由来のノイズを付与するモジュール
3つのパターンのノイズ生成関数を提供
"""

import torch
import numpy as np


def add_frequency_band_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, band_ratio=0.3):
    """
    パターン1: 周波数帯域集中ノイズ
    特定の周波数帯域に集中的に発生するノイズ
    電源ノイズ、共振、クロストークなどを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        band_ratio: ノイズが集中する帯域の割合（0.3 = 30%のポイントに集中的にノイズ）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # ノイズが集中する帯域の幅を計算
    band_width = int(points_per_interval * band_ratio)
    
    # 帯域の中心位置をランダムに決定（帯域が区間内に収まるように）
    max_center = points_per_interval - band_width // 2
    min_center = band_width // 2
    band_center = torch.randint(min_center, max_center, (1,)).item()
    
    # ガウシアン分布でノイズの強度を減衰させる
    positions = torch.arange(points_per_interval, dtype=torch.float32)
    # 帯域の中心からの距離
    distances = torch.abs(positions - band_center)
    # ガウシアン分布の標準偏差（帯域幅の1/3程度）
    sigma = band_width / 3.0
    # ガウシアン重み（中心で1.0、外側で減衰）
    weights = torch.exp(-0.5 * (distances / sigma) ** 2)
    
    # ベースノイズを生成
    base_noise = torch.randn(points_per_interval) * interval_mean * noise_level
    
    # ガウシアン重みをかけて帯域集中ノイズを生成
    noise = base_noise * weights
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx


def add_localized_spike_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, spike_ratio=0.15):
    """
    パターン2: 局所スパイクノイズ
    一部のポイントに集中的に発生するスパイク状のノイズ
    電磁干渉（EMI）、静電気放電（ESD）、接触不良などを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        spike_ratio: スパイクが発生するポイントの割合（0.15 = 15%のポイントに大きなスパイク）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # スパイクが発生するポイント数を計算
    num_spike_points = int(points_per_interval * spike_ratio)
    num_spike_points = max(1, num_spike_points)  # 最低1ポイントはスパイク
    
    # スパイクが発生するポイントをランダムに選択
    spike_indices = torch.randperm(points_per_interval)[:num_spike_points]
    
    # ノイズを初期化
    noise = torch.zeros(points_per_interval)
    
    # スパイクポイントには大きなノイズ（3-5倍の強度）
    spike_strength_multiplier = 3.0 + torch.rand(num_spike_points) * 2.0  # 3.0-5.0の範囲
    noise[spike_indices] = torch.randn(num_spike_points) * interval_mean * noise_level * spike_strength_multiplier
    
    # その他のポイントには小さなバックグラウンドノイズ（0.1倍の強度）
    other_indices = torch.ones(points_per_interval, dtype=torch.bool)
    other_indices[spike_indices] = False
    noise[other_indices] = torch.randn(points_per_interval - num_spike_points) * interval_mean * noise_level * 0.1
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx


def add_amplitude_dependent_noise(psd_data, interval_idx, noise_level=0.1, num_intervals=30, top_ratio=0.3):
    """
    パターン3: 振幅依存ノイズ
    元の信号が大きい領域に集中的にノイズが発生
    非線形増幅器の歪み、ADCの量子化ノイズ、飽和などを模擬
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス
        noise_level: ノイズの強度（元の値に対する倍率）
        num_intervals: 区間数
        top_ratio: ノイズが集中する上位の割合（0.3 = 上位30%のポイントに集中的にノイズ）
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    noisy_data = psd_data.clone()
    points_per_interval = len(psd_data) // num_intervals
    
    start_idx = interval_idx * points_per_interval
    end_idx = start_idx + points_per_interval
    
    # その区間のデータを取得
    interval_data = psd_data[start_idx:end_idx]
    
    # 振幅の大きい順にソートしてインデックスを取得
    sorted_indices = torch.argsort(interval_data, descending=True)
    
    # 上位のポイント数を計算
    num_top_points = int(points_per_interval * top_ratio)
    num_top_points = max(1, num_top_points)  # 最低1ポイント
    
    # 上位のポイントのインデックス
    top_indices = sorted_indices[:num_top_points]
    
    # ノイズを初期化
    noise = torch.zeros(points_per_interval)
    
    # 上位ポイントには大きなノイズ（振幅に比例）
    # 各ポイントの振幅に応じてノイズの強度を変える
    top_values = interval_data[top_indices]
    top_mean = interval_data.mean()
    
    # 振幅が大きいほどノイズが大きくなる（最大2倍まで）
    amplitude_factors = 1.0 + (top_values / (top_mean + 1e-10))
    amplitude_factors = torch.clamp(amplitude_factors, 1.0, 2.0)
    
    noise[top_indices] = torch.randn(num_top_points) * top_mean * noise_level * amplitude_factors
    
    # その他のポイントには小さなノイズ（0.2倍の強度）
    other_indices = torch.ones(points_per_interval, dtype=torch.bool)
    other_indices[top_indices] = False
    noise[other_indices] = torch.randn(points_per_interval - num_top_points) * top_mean * noise_level * 0.2
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx


def add_noise_to_interval(psd_data, interval_idx, noise_type='frequency_band', **kwargs):
    """
    統一インターフェース: ノイズタイプを指定してノイズを付与
    
    Args:
        psd_data: PSDデータ
        interval_idx: ノイズを加える区間のインデックス
        noise_type: 'frequency_band', 'localized_spike', 'amplitude_dependent' のいずれか
        **kwargs: 各ノイズ関数への追加パラメータ
    
    Returns:
        ノイズを加えたPSDデータ, 開始インデックス, 終了インデックス
    """
    if noise_type == 'frequency_band':
        return add_frequency_band_noise(psd_data, interval_idx, **kwargs)
    elif noise_type == 'localized_spike':
        return add_localized_spike_noise(psd_data, interval_idx, **kwargs)
    elif noise_type == 'amplitude_dependent':
        return add_amplitude_dependent_noise(psd_data, interval_idx, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}. Choose from 'frequency_band', 'localized_spike', 'amplitude_dependent'")
