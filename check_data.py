"""
PSDデータの確認用スクリプト
data_lowF_noise.pickleの中身を確認します
"""

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ
y = data['y']  # おそらく活性化エネルギー（使わない）

print("=" * 60)
print("データの基本情報")
print("=" * 60)
print(f"サンプル数: {x.shape[0]:,}")
print(f"PSDデータ形状: {x.shape[1:]} (チャンネル数 x 周波数ポイント数)")
print(f"各PSDデータの長さ: {x.shape[2]:,}ポイント")
print(f"\nyの形状: {y.shape}")
print(f"yの値の範囲: [{y[:, 0].min():.2f}, {y[:, 0].max():.2f}], [{y[:, 1].min():.2f}, {y[:, 1].max():.2f}]")

print("\n" + "=" * 60)
print("PSDデータの統計（最初の5サンプル）")
print("=" * 60)
for i in range(min(5, x.shape[0])):
    sample_psd = x[i, 0, :].numpy()
    print(f"\nサンプル {i+1}:")
    print(f"  最小値: {sample_psd.min():.6e}")
    print(f"  最大値: {sample_psd.max():.6e}")
    print(f"  平均値: {sample_psd.mean():.6e}")
    print(f"  中央値: {np.median(sample_psd):.6e}")

print("\n" + "=" * 60)
print("PSDデータの可視化（最初の3サンプル）")
print("=" * 60)

# 最初の3サンプルをプロット
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
for i in range(3):
    sample_psd = x[i, 0, :].numpy()
    axes[i].plot(sample_psd)
    axes[i].set_title(f'サンプル {i+1} のPSDデータ')
    axes[i].set_xlabel('周波数ポイント')
    axes[i].set_ylabel('PSD値')
    axes[i].set_yscale('log')  # 対数スケールで表示
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('psd_samples.png', dpi=150)
print("グラフを 'psd_samples.png' に保存しました")

# 全サンプルの平均PSDもプロット
print("\n全サンプルの平均PSDを計算中...")
mean_psd = x[:, 0, :].mean(dim=0).numpy()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(mean_psd)
ax.set_title('全32000サンプルの平均PSD')
ax.set_xlabel('周波数ポイント')
ax.set_ylabel('PSD値（平均）')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mean_psd.png', dpi=150)
print("平均PSDグラフを 'mean_psd.png' に保存しました")

print("\nデータ確認完了！")


