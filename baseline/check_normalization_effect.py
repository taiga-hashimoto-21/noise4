"""
正規化の影響を確認するスクリプト
正規化前後でノイズがどのように変化するかを可視化
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# データセットの読み込み
print("データセットを読み込み中...")
with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['train']['data']
train_labels = dataset['train']['labels']

# サンプルを取得（各クラスから1つずつ）
samples = []
sample_labels = []
for i in range(10):
    indices = np.where(np.array(train_labels) == i)[0]
    if len(indices) > 0:
        sample = train_data[indices[0]]
        # Tensorの場合はnumpy配列に変換
        if hasattr(sample, 'numpy'):
            sample = sample.numpy()
        elif hasattr(sample, 'cpu'):
            sample = sample.cpu().numpy()
        samples.append(sample)
        sample_labels.append(i)

samples = np.array(samples)
sample_labels = np.array(sample_labels)

# 正規化前のデータ
data_before = samples.copy()

# 正規化後のデータ
# train_dataをnumpy配列に変換
train_data_np = train_data
if hasattr(train_data, 'numpy'):
    train_data_np = train_data.numpy()
elif hasattr(train_data, 'cpu'):
    train_data_np = train_data.cpu().numpy()

train_mean = np.mean(train_data_np)
train_std = np.std(train_data_np)
data_after = (samples - train_mean) / (train_std + 1e-8)

# ノイズ区間を計算（各クラスに対応）
noise_intervals = []
for label in sample_labels:
    # 各クラスは300ポイントの区間に対応
    start_idx = label * 300
    end_idx = start_idx + 300
    noise_intervals.append((start_idx, end_idx))

# 可視化
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('正規化前後の比較（各クラスのサンプル）', fontsize=16)

for i in range(10):
    row = i // 5
    col = i % 5
    ax = axes[row, col]
    
    # 正規化前
    ax.plot(data_before[i], label='正規化前', alpha=0.7, linewidth=1)
    
    # 正規化後
    ax.plot(data_after[i], label='正規化後', alpha=0.7, linewidth=1)
    
    # ノイズ区間を強調
    start_idx, end_idx = noise_intervals[i]
    ax.axvspan(start_idx, end_idx, alpha=0.2, color='red', label='ノイズ区間')
    
    ax.set_title(f'クラス {i} (ノイズ区間: {start_idx}-{end_idx})')
    ax.set_xlabel('ポイント')
    ax.set_ylabel('値')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 正規化前後の比較を 'normalization_comparison.png' に保存しました")

# 統計情報を表示
print("\n=== 正規化前の統計 ===")
print(f"平均: {data_before.mean():.6e}")
print(f"標準偏差: {data_before.std():.6e}")
print(f"最小値: {data_before.min():.6e}")
print(f"最大値: {data_before.max():.6e}")

print("\n=== 正規化後の統計 ===")
print(f"平均: {data_after.mean():.6e}")
print(f"標準偏差: {data_after.std():.6e}")
print(f"最小値: {data_after.min():.6e}")
print(f"最大値: {data_after.max():.6e}")

# ノイズ区間と他の区間の差を計算
print("\n=== ノイズ区間と他の区間の差（正規化前） ===")
for i, label in enumerate(sample_labels):
    start_idx, end_idx = noise_intervals[i]
    noise_region = data_before[i, start_idx:end_idx]
    other_regions = np.concatenate([
        data_before[i, :start_idx],
        data_before[i, end_idx:]
    ])
    
    noise_mean = noise_region.mean()
    other_mean = other_regions.mean()
    diff = abs(noise_mean - other_mean)
    ratio = noise_mean / other_mean if other_mean != 0 else 0
    
    print(f"クラス {label}: ノイズ平均={noise_mean:.6e}, 他平均={other_mean:.6e}, 差={diff:.6e}, 比={ratio:.2f}x")

print("\n=== ノイズ区間と他の区間の差（正規化後） ===")
for i, label in enumerate(sample_labels):
    start_idx, end_idx = noise_intervals[i]
    noise_region = data_after[i, start_idx:end_idx]
    other_regions = np.concatenate([
        data_after[i, :start_idx],
        data_after[i, end_idx:]
    ])
    
    noise_mean = noise_region.mean()
    other_mean = other_regions.mean()
    diff = abs(noise_mean - other_mean)
    ratio = noise_mean / other_mean if other_mean != 0 else 0
    
    print(f"クラス {label}: ノイズ平均={noise_mean:.6e}, 他平均={other_mean:.6e}, 差={diff:.6e}, 比={ratio:.2f}x")

plt.show()

