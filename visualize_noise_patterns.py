"""
3パターンのノイズ付与方法を可視化するスクリプト
測定系由来のノイズの違いを比較
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('.')
from noise.add_noise import add_gaussian_noise, add_pink_noise, add_burst_noise

# 日本語フォントの設定（Macの場合）
import platform
if platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'Hiragino Sans'  # または 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化けを防ぐ

# データの読み込み
print("データを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ

# パラメータ設定
NUM_INTERVALS = 30
NOISE_INTERVAL_START = 10
NOISE_INTERVAL_END = 20
NOISE_LEVEL = 0.5  # 発表用に適切なノイズレベル

print(f"区間数: {NUM_INTERVALS}")
print(f"ノイズ付与範囲: 区間 {NOISE_INTERVAL_START+1} ~ {NOISE_INTERVAL_END+1}")
print(f"ノイズレベル: {NOISE_LEVEL}")

# 同じサンプル、同じ区間に3パターンのノイズを付与
sample_idx = 0
noise_interval = random.randint(NOISE_INTERVAL_START, NOISE_INTERVAL_END)
original_psd = x[sample_idx, 0, :]

print(f"\nサンプル {sample_idx+1}, 区間 {noise_interval+1} にノイズを付与")

# 3パターンのノイズを付与
gaussian_psd, g_start, g_end = add_gaussian_noise(
    original_psd, noise_interval, noise_level=NOISE_LEVEL, num_intervals=NUM_INTERVALS
)
pink_psd, p_start, p_end = add_pink_noise(
    original_psd, noise_interval, noise_level=NOISE_LEVEL, num_intervals=NUM_INTERVALS
)
burst_psd, b_start, b_end = add_burst_noise(
    original_psd, noise_interval, noise_level=NOISE_LEVEL, num_intervals=NUM_INTERVALS
)

# 可視化: 全体の比較（発表用にサイズとフォントを調整）
plt.rcParams['font.size'] = 12  # フォントサイズを大きく
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 元のデータ
axes[0].plot(original_psd.numpy(), label='元データ（ノイズなし）', color='black', linewidth=2.5)
axes[0].axvspan(g_start, g_end, alpha=0.2, color='gray', label='ノイズ付与領域')
axes[0].set_title('元のPSDデータ（ノイズなし）', fontsize=16, fontweight='bold')
axes[0].set_xlabel('周波数ポイント', fontsize=14)
axes[0].set_ylabel('PSD値', fontsize=14)
axes[0].set_yscale('log')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# パターン1: ガウシアンノイズ
axes[1].plot(original_psd.numpy(), label='元データ', color='black', alpha=0.5, linewidth=2)
axes[1].plot(gaussian_psd.numpy(), label='ホワイトノイズ付与後', color='blue', linewidth=2.5)
axes[1].axvspan(g_start, g_end, alpha=0.2, color='red', label='ノイズ付与領域')
axes[1].set_title('パターン1：ホワイトノイズ（ガウシアンノイズ）\n測定器の熱雑音を模擬', 
                  fontsize=16, fontweight='bold')
axes[1].set_xlabel('周波数ポイント', fontsize=14)
axes[1].set_ylabel('PSD値', fontsize=14)
axes[1].set_yscale('log')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

# パターン2: ピンクノイズ
axes[2].plot(original_psd.numpy(), label='元データ', color='black', alpha=0.5, linewidth=2)
axes[2].plot(pink_psd.numpy(), label='1/fノイズ付与後', color='green', linewidth=2.5)
axes[2].axvspan(p_start, p_end, alpha=0.2, color='red', label='ノイズ付与領域')
axes[2].set_title('パターン2：1/fノイズ（ピンクノイズ）\n測定器のドリフトやフリッカーノイズを模擬', 
                  fontsize=16, fontweight='bold')
axes[2].set_xlabel('周波数ポイント', fontsize=14)
axes[2].set_ylabel('PSD値', fontsize=14)
axes[2].set_yscale('log')
axes[2].legend(fontsize=12)
axes[2].grid(True, alpha=0.3)

# パターン3: バーストノイズ
axes[3].plot(original_psd.numpy(), label='元データ', color='black', alpha=0.5, linewidth=2)
axes[3].plot(burst_psd.numpy(), label='バーストノイズ付与後', color='red', linewidth=2.5)
axes[3].axvspan(b_start, b_end, alpha=0.2, color='red', label='ノイズ付与領域')
axes[3].set_title('パターン3：バーストノイズ（スパイクノイズ）\n電磁干渉や外乱を模擬', 
                  fontsize=16, fontweight='bold')
axes[3].set_xlabel('周波数ポイント', fontsize=14)
axes[3].set_ylabel('PSD値', fontsize=14)
axes[3].set_yscale('log')
axes[3].legend(fontsize=12)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_patterns_comparison.png', dpi=300, bbox_inches='tight')  # 解像度を上げる
print("\n全体比較グラフを 'noise_patterns_comparison.png' に保存しました（発表用：300dpi）")

# 拡大表示: ノイズが加えられた区間を拡大
zoom_start = max(0, g_start - 50)
zoom_end = min(3000, g_end + 50)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# パターン1: ガウシアンノイズ（拡大）
axes[0].plot(original_psd[zoom_start:zoom_end].numpy(), label='元データ', 
             color='black', alpha=0.7, linewidth=2, marker='o', markersize=4)
axes[0].plot(gaussian_psd[zoom_start:zoom_end].numpy(), label='ホワイトノイズ付与後', 
             color='blue', linewidth=2, marker='s', markersize=4)
axes[0].axvspan(g_start - zoom_start, g_end - zoom_start, alpha=0.2, color='red', label='ノイズ付与領域')
axes[0].set_title('パターン1：ホワイトノイズ（拡大）', fontsize=14, fontweight='bold')
axes[0].set_xlabel('周波数ポイント（相対）')
axes[0].set_ylabel('PSD値')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# パターン2: ピンクノイズ（拡大）
axes[1].plot(original_psd[zoom_start:zoom_end].numpy(), label='元データ', 
             color='black', alpha=0.7, linewidth=2, marker='o', markersize=4)
axes[1].plot(pink_psd[zoom_start:zoom_end].numpy(), label='1/fノイズ付与後', 
             color='green', linewidth=2, marker='s', markersize=4)
axes[1].axvspan(p_start - zoom_start, p_end - zoom_start, alpha=0.2, color='red', label='ノイズ付与領域')
axes[1].set_title('パターン2：1/fノイズ（拡大）\n低周波数ほどノイズが大きくなる', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('周波数ポイント（相対）')
axes[1].set_ylabel('PSD値')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# パターン3: バーストノイズ（拡大）
axes[2].plot(original_psd[zoom_start:zoom_end].numpy(), label='元データ', 
             color='black', alpha=0.7, linewidth=2, marker='o', markersize=4)
axes[2].plot(burst_psd[zoom_start:zoom_end].numpy(), label='バーストノイズ付与後', 
             color='red', linewidth=2, marker='s', markersize=4)
axes[2].axvspan(b_start - zoom_start, b_end - zoom_start, alpha=0.2, color='red', label='ノイズ付与領域')
axes[2].set_title('パターン3：バーストノイズ（拡大）\nランダムなポイントにスパイク状のノイズが発生', 
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('周波数ポイント（相対）')
axes[2].set_ylabel('PSD値')
axes[2].set_yscale('log')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_patterns_zoomed.png', dpi=150, bbox_inches='tight')
print("拡大比較グラフを 'noise_patterns_zoomed.png' に保存しました")

# ノイズの差分を可視化（ノイズだけを表示）
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

noise_region_start = g_start
noise_region_end = g_end

# パターン1のノイズ差分
gaussian_noise_only = (gaussian_psd - original_psd)[noise_region_start:noise_region_end]
axes[0].plot(gaussian_noise_only.numpy(), color='blue', linewidth=2)
axes[0].set_title('パターン1：ホワイトノイズのみ\nすべての周波数で一様なノイズ', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('周波数ポイント（ノイズ領域内）')
axes[0].set_ylabel('ノイズ振幅')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# パターン2のノイズ差分
pink_noise_only = (pink_psd - original_psd)[noise_region_start:noise_region_end]
axes[1].plot(pink_noise_only.numpy(), color='green', linewidth=2)
axes[1].set_title('パターン2：1/fノイズのみ\n周波数が高いほどノイズが小さくなる', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('周波数ポイント（ノイズ領域内）')
axes[1].set_ylabel('ノイズ振幅')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# パターン3のノイズ差分
burst_noise_only = (burst_psd - original_psd)[noise_region_start:noise_region_end]
axes[2].plot(burst_noise_only.numpy(), color='red', linewidth=2, marker='o', markersize=3)
axes[2].set_title('パターン3：バーストノイズのみ\nランダムなポイントにスパイク状のノイズ', 
                  fontsize=14, fontweight='bold')
axes[2].set_xlabel('周波数ポイント（ノイズ領域内）')
axes[2].set_ylabel('ノイズ振幅')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('noise_patterns_difference.png', dpi=150, bbox_inches='tight')
print("ノイズ差分グラフを 'noise_patterns_difference.png' に保存しました")

print("\n=== ノイズパターンの説明 ===")
print("1. Gaussian Noise (ホワイトノイズ):")
print("   - すべての周波数で一様なノイズ")
print("   - 測定器の熱雑音や電子回路の熱雑音を模擬")
print("   - 最も一般的なノイズ")
print("\n2. Pink Noise (1/fノイズ):")
print("   - 低周波数ほど大きくなるノイズ")
print("   - 測定器のドリフトや環境変動を模擬")
print("   - フリッカーノイズとも呼ばれる")
print("\n3. Burst Noise (バーストノイズ):")
print("   - 特定のポイントに集中的に発生するスパイク状のノイズ")
print("   - 電磁干渉や外乱を模擬")
print("   - 30%のポイントに大きなノイズ、残りは小さなノイズ")

print("\n可視化完了！")

