"""
ノイズ付与前後のPSDデータを可視化するスクリプト
30区間に分割して、ランダムに1つの区間にノイズを加える
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# 日本語フォントの設定（Macの場合）
plt.rcParams['font.family'] = 'DejaVu Sans'

# データの読み込み
print("データを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ

# パラメータ設定
NUM_INTERVALS = 30  # 30区間
POINTS_PER_INTERVAL = 3000 // NUM_INTERVALS  # 1区間あたり100ポイント
# ノイズを付与する区間の範囲（真ん中あたり）
NOISE_INTERVAL_START = 10  # 10区間目から
NOISE_INTERVAL_END = 20    # 20区間目まで
print(f"区間数: {NUM_INTERVALS}")
print(f"1区間あたりのポイント数: {POINTS_PER_INTERVAL}")
print(f"ノイズ付与範囲: 区間 {NOISE_INTERVAL_START+1} ~ {NOISE_INTERVAL_END+1} (真ん中あたり)")

def add_noise_to_interval(psd_data, interval_idx, noise_level=0.1):
    """
    指定した区間にノイズを加える
    
    Args:
        psd_data: PSDデータ (3000ポイント)
        interval_idx: ノイズを加える区間のインデックス (0~29)
        noise_level: ノイズの強度（元の値に対する倍率）
    
    Returns:
        ノイズを加えたPSDデータ
    """
    noisy_data = psd_data.clone()
    
    # 区間の開始位置と終了位置を計算
    start_idx = interval_idx * POINTS_PER_INTERVAL
    end_idx = start_idx + POINTS_PER_INTERVAL
    
    # その区間の平均値を取得
    interval_mean = psd_data[start_idx:end_idx].mean()
    
    # ノイズを生成（ガウシアンノイズ）
    noise = torch.randn(POINTS_PER_INTERVAL) * interval_mean * noise_level
    
    # ノイズを加える
    noisy_data[start_idx:end_idx] += noise
    
    return noisy_data, start_idx, end_idx

# 最初の5サンプルでノイズ付与前後を可視化
num_samples = 5
fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3*num_samples))

for i in range(num_samples):
    # 元のPSDデータ
    original_psd = x[i, 0, :]
    
    # ランダムに1つの区間を選んでノイズを加える（真ん中あたり）
    noise_interval = random.randint(NOISE_INTERVAL_START, NOISE_INTERVAL_END)
    noisy_psd, start_idx, end_idx = add_noise_to_interval(
        original_psd, 
        noise_interval,
        noise_level=0.2  # ノイズレベル（調整可能）
    )
    
    # プロット
    axes[i].plot(original_psd.numpy(), label='Original (no noise)', alpha=0.7, linewidth=1.5)
    axes[i].plot(noisy_psd.numpy(), label=f'With noise (interval {noise_interval+1})', alpha=0.7, linewidth=1.5)
    
    # ノイズが加えられた区間をハイライト
    axes[i].axvspan(start_idx, end_idx, alpha=0.2, color='red', label='Noise region')
    
    axes[i].set_title(f'Sample {i+1}: Noise added to interval {noise_interval+1} ({start_idx}-{end_idx})')
    axes[i].set_xlabel('Frequency point')
    axes[i].set_ylabel('PSD value')
    axes[i].set_yscale('log')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_comparison.png', dpi=150, bbox_inches='tight')
print("\nグラフを 'noise_comparison.png' に保存しました")

# 1つのサンプルを拡大して表示
print("\n1つのサンプルを拡大表示中...")
sample_idx = 0
original_psd = x[sample_idx, 0, :]
noise_interval = random.randint(NOISE_INTERVAL_START, NOISE_INTERVAL_END)
noisy_psd, start_idx, end_idx = add_noise_to_interval(original_psd, noise_interval, noise_level=0.2)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 全体のグラフ
axes[0].plot(original_psd.numpy(), label='Original (no noise)', alpha=0.7, linewidth=2)
axes[0].plot(noisy_psd.numpy(), label=f'With noise (interval {noise_interval+1})', alpha=0.7, linewidth=2)
axes[0].axvspan(start_idx, end_idx, alpha=0.2, color='red', label='Noise region')
axes[0].set_title(f'Sample {sample_idx+1}: Full PSD (3000 points) - Noise in interval {noise_interval+1}')
axes[0].set_xlabel('Frequency point')
axes[0].set_ylabel('PSD value (log scale)')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ノイズが加えられた区間を拡大
zoom_start = max(0, start_idx - 50)
zoom_end = min(3000, end_idx + 50)
axes[1].plot(original_psd[zoom_start:zoom_end].numpy(), label='Original', alpha=0.7, linewidth=2, marker='o', markersize=3)
axes[1].plot(noisy_psd[zoom_start:zoom_end].numpy(), label='With noise', alpha=0.7, linewidth=2, marker='s', markersize=3)
axes[1].axvspan(start_idx - zoom_start, end_idx - zoom_start, alpha=0.2, color='red', label='Noise region')
axes[1].set_title(f'Zoomed view: Interval {noise_interval+1} (points {start_idx}-{end_idx})')
axes[1].set_xlabel('Frequency point (relative)')
axes[1].set_ylabel('PSD value')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_comparison_zoomed.png', dpi=150, bbox_inches='tight')
print("拡大グラフを 'noise_comparison_zoomed.png' に保存しました")

print("\n可視化完了！")

