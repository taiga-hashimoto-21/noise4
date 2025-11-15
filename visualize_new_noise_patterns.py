"""
新しい3パターンのノイズ付与方法を可視化するスクリプト
測定系由来のノイズの違いを比較
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
sys.path.append('.')
from noise.add_noise import add_noise_to_interval

# 日本語フォントの設定
import platform
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# データの読み込み
print("データを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ

# パラメータ設定
NUM_INTERVALS = 10
NOISE_LEVEL = 0.3  # ノイズレベル（元の値の30%程度）

print(f"区間数: {NUM_INTERVALS}")
print(f"ノイズレベル: {NOISE_LEVEL}")

# 同じサンプル、同じ区間に3パターンのノイズを付与
sample_idx = 0
noise_interval = random.randint(0, NUM_INTERVALS - 1)
original_psd = x[sample_idx, 0, :]

print(f"\nサンプル {sample_idx+1}, 区間 {noise_interval+1} にノイズを付与")

# 3パターンのノイズを付与
frequency_band_psd, fb_start, fb_end = add_noise_to_interval(
    original_psd, noise_interval, 
    noise_type='frequency_band',
    noise_level=NOISE_LEVEL, 
    num_intervals=NUM_INTERVALS
)

localized_spike_psd, ls_start, ls_end = add_noise_to_interval(
    original_psd, noise_interval, 
    noise_type='localized_spike',
    noise_level=NOISE_LEVEL, 
    num_intervals=NUM_INTERVALS
)

amplitude_dependent_psd, ad_start, ad_end = add_noise_to_interval(
    original_psd, noise_interval, 
    noise_type='amplitude_dependent',
    noise_level=NOISE_LEVEL, 
    num_intervals=NUM_INTERVALS
)

# 可視化: 1つのサンプルに同じ区間で異なるノイズタイプを付与した比較（添付画像と同じ形式）
plt.rcParams['font.size'] = 12
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# すべて同じ区間を使用（fb_start, fb_end）
noise_start = fb_start
noise_end = fb_end

# 1つ目: 元のデータ（ノイズなし）
axes[0].plot(original_psd.numpy(), label='元データ（ノイズなし）', color='black', linewidth=2.5)
axes[0].axvspan(noise_start, noise_end, alpha=0.2, color='gray', label='ノイズ付与領域')
axes[0].set_title('元のPSDデータ（ノイズなし）', fontsize=16, fontweight='bold')
axes[0].set_xlabel('周波数ポイント', fontsize=14)
axes[0].set_ylabel('PSD値', fontsize=14)
axes[0].set_yscale('log')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# 2つ目: 周波数帯域集中ノイズを1区間に付与
axes[1].plot(original_psd.numpy(), label='元データ', color='black', alpha=0.5, linewidth=2)
axes[1].plot(frequency_band_psd.numpy(), label='周波数帯域集中ノイズ付与後', color='blue', linewidth=2.5)
axes[1].axvspan(noise_start, noise_end, alpha=0.2, color='red', label='ノイズ付与領域')
axes[1].set_title('パターン1：周波数帯域集中ノイズ（電源ノイズ、共振、クロストークを模擬）', 
                  fontsize=16, fontweight='bold')
axes[1].set_xlabel('周波数ポイント', fontsize=14)
axes[1].set_ylabel('PSD値', fontsize=14)
axes[1].set_yscale('log')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

# 3つ目: 局所スパイクノイズを1区間に付与
axes[2].plot(original_psd.numpy(), label='元データ', color='black', alpha=0.5, linewidth=2)
axes[2].plot(localized_spike_psd.numpy(), label='局所スパイクノイズ付与後', color='green', linewidth=2.5)
axes[2].axvspan(noise_start, noise_end, alpha=0.2, color='red', label='ノイズ付与領域')
axes[2].set_title('パターン2：局所スパイクノイズ（電磁干渉（EMI）、静電気放電（ESD）、接触不良を模擬）', 
                  fontsize=16, fontweight='bold')
axes[2].set_xlabel('周波数ポイント', fontsize=14)
axes[2].set_ylabel('PSD値', fontsize=14)
axes[2].set_yscale('log')
axes[2].legend(fontsize=12)
axes[2].grid(True, alpha=0.3)

# 4つ目: 振幅依存ノイズを1区間に付与
axes[3].plot(original_psd.numpy(), label='元データ', color='black', alpha=0.5, linewidth=2)
axes[3].plot(amplitude_dependent_psd.numpy(), label='振幅依存ノイズ付与後', color='red', linewidth=2.5)
axes[3].axvspan(noise_start, noise_end, alpha=0.2, color='red', label='ノイズ付与領域')
axes[3].set_title('パターン3：振幅依存ノイズ（非線形増幅器の歪み、ADCの量子化ノイズ、飽和を模擬）', 
                  fontsize=16, fontweight='bold')
axes[3].set_xlabel('周波数ポイント', fontsize=14)
axes[3].set_ylabel('PSD値', fontsize=14)
axes[3].set_yscale('log')
axes[3].legend(fontsize=12)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'new_noise_patterns_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n全体比較グラフを '{output_path}' に保存しました（300dpi）")
print(f"  サンプル: {sample_idx+1}, 区間: {noise_interval+1} ({noise_start}-{noise_end}ポイント)")

# 拡大表示: ノイズが加えられた区間を拡大
zoom_start = max(0, fb_start - 50)
zoom_end = min(3000, fb_end + 50)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# パターン1: 周波数帯域集中ノイズ（拡大）
axes[0].plot(original_psd[zoom_start:zoom_end].numpy(), label='元データ', 
             color='black', alpha=0.7, linewidth=2, marker='o', markersize=4)
axes[0].plot(frequency_band_psd[zoom_start:zoom_end].numpy(), label='周波数帯域集中ノイズ付与後', 
             color='blue', linewidth=2, marker='s', markersize=4)
axes[0].axvspan(fb_start - zoom_start, fb_end - zoom_start, alpha=0.2, color='red', label='ノイズ付与領域')
axes[0].set_title('パターン1：周波数帯域集中ノイズ（拡大）\n特定の周波数帯域にガウシアン分布でノイズが集中', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('周波数ポイント（相対）')
axes[0].set_ylabel('PSD値')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# パターン2: 局所スパイクノイズ（拡大）
axes[1].plot(original_psd[zoom_start:zoom_end].numpy(), label='元データ', 
             color='black', alpha=0.7, linewidth=2, marker='o', markersize=4)
axes[1].plot(localized_spike_psd[zoom_start:zoom_end].numpy(), label='局所スパイクノイズ付与後', 
             color='green', linewidth=2, marker='s', markersize=4)
axes[1].axvspan(ls_start - zoom_start, ls_end - zoom_start, alpha=0.2, color='red', label='ノイズ付与領域')
axes[1].set_title('パターン2：局所スパイクノイズ（拡大）\n一部のポイント（15%）に大きなスパイク、残りは小さいバックグラウンドノイズ', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('周波数ポイント（相対）')
axes[1].set_ylabel('PSD値')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# パターン3: 振幅依存ノイズ（拡大）
axes[2].plot(original_psd[zoom_start:zoom_end].numpy(), label='元データ', 
             color='black', alpha=0.7, linewidth=2, marker='o', markersize=4)
axes[2].plot(amplitude_dependent_psd[zoom_start:zoom_end].numpy(), label='振幅依存ノイズ付与後', 
             color='red', linewidth=2, marker='s', markersize=4)
axes[2].axvspan(ad_start - zoom_start, ad_end - zoom_start, alpha=0.2, color='red', label='ノイズ付与領域')
axes[2].set_title('パターン3：振幅依存ノイズ（拡大）\n元の信号が大きい領域（上位30%）に集中的にノイズ', 
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('周波数ポイント（相対）')
axes[2].set_ylabel('PSD値')
axes[2].set_yscale('log')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_path_zoom = 'new_noise_patterns_zoomed.png'
plt.savefig(output_path_zoom, dpi=150, bbox_inches='tight')
print(f"拡大比較グラフを '{output_path_zoom}' に保存しました")

# ノイズの差分を可視化（ノイズだけを表示）
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

noise_region_start = fb_start
noise_region_end = fb_end

# パターン1のノイズ差分
frequency_band_noise_only = (frequency_band_psd - original_psd)[noise_region_start:noise_region_end]
axes[0].plot(frequency_band_noise_only.numpy(), color='blue', linewidth=2)
axes[0].set_title('パターン1：周波数帯域集中ノイズのみ\n特定の周波数帯域にガウシアン分布でノイズが集中', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('周波数ポイント（ノイズ領域内）')
axes[0].set_ylabel('ノイズ振幅')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# パターン2のノイズ差分
localized_spike_noise_only = (localized_spike_psd - original_psd)[noise_region_start:noise_region_end]
axes[1].plot(localized_spike_noise_only.numpy(), color='green', linewidth=2, marker='o', markersize=3)
axes[1].set_title('パターン2：局所スパイクノイズのみ\n一部のポイントに大きなスパイク、残りは小さいバックグラウンドノイズ', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('周波数ポイント（ノイズ領域内）')
axes[1].set_ylabel('ノイズ振幅')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# パターン3のノイズ差分
amplitude_dependent_noise_only = (amplitude_dependent_psd - original_psd)[noise_region_start:noise_region_end]
axes[2].plot(amplitude_dependent_noise_only.numpy(), color='red', linewidth=2)
axes[2].set_title('パターン3：振幅依存ノイズのみ\n元の信号が大きい領域に集中的にノイズ', 
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('周波数ポイント（ノイズ領域内）')
axes[2].set_ylabel('ノイズ振幅')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
output_path_diff = 'new_noise_patterns_difference.png'
plt.savefig(output_path_diff, dpi=150, bbox_inches='tight')
print(f"ノイズ差分グラフを '{output_path_diff}' に保存しました")

print("\n=== ノイズパターンの説明 ===")
print("1. Frequency Band Noise (周波数帯域集中ノイズ):")
print("   - 特定の周波数帯域にガウシアン分布でノイズが集中")
print("   - 電源ノイズ、共振、クロストークを模擬")
print("   - 区間内の30%程度の帯域に集中的に発生")
print("\n2. Localized Spike Noise (局所スパイクノイズ):")
print("   - 一部のポイント（15%）に大きなスパイク")
print("   - 残りのポイントには小さなバックグラウンドノイズ")
print("   - 電磁干渉（EMI）、静電気放電（ESD）、接触不良を模擬")
print("\n3. Amplitude Dependent Noise (振幅依存ノイズ):")
print("   - 元の信号が大きい領域（上位30%）に集中的にノイズ")
print("   - 非線形増幅器の歪み、ADCの量子化ノイズ、飽和を模擬")
print("   - 信号レベルに応じてノイズの強度が変化")

print("\n可視化完了！")
print(f"生成された画像:")
print(f"  - {output_path}")
print(f"  - {output_path_zoom}")
print(f"  - {output_path_diff}")

