"""
10クラス分類用のデータセット準備
32000点すべてにノイズを付与し、ラベルを作成

ノイズタイプ:
- 'frequency_band': 周波数帯域集中ノイズ（特定の周波数帯域に集中的に発生）
- 'localized_spike': 局所スパイクノイズ（一部のポイントに集中的に発生）
- 'amplitude_dependent': 振幅依存ノイズ（信号が大きい領域に集中的に発生）
"""

import pickle
import torch
import numpy as np
import random
from noise.add_noise import add_noise_to_interval

# パラメータ設定
NUM_INTERVALS = 10  # 30 → 10に変更（10クラス分類）
NOISE_LEVEL = 0.3  # ノイズレベル（元の値の30%程度）
NOISE_TYPE = 'frequency_band'  # 'frequency_band', 'localized_spike', 'amplitude_dependent' から選択

print("=" * 60)
print("データセットの準備")
print("=" * 60)
print(f"区間数: {NUM_INTERVALS}")
print(f"ノイズタイプ: {NOISE_TYPE}")
print(f"ノイズレベル: {NOISE_LEVEL}")
print(f"ノイズ付与範囲: 1~{NUM_INTERVALS}区間（全範囲）")

# データの読み込み
print("\nデータを読み込み中...")
with open('data_lowF_noise.pickle', 'rb') as f:
    data = pickle.load(f)

x = data['x']  # PSDデータ (32000, 1, 3000)
num_samples = x.shape[0]

print(f"サンプル数: {num_samples:,}")

# データセットの準備
print("\nノイズを付与中...")
noisy_data = []
labels = []

for i in range(num_samples):
    # 元のPSDデータ
    original_psd = x[i, 0, :]
    
    # ランダムに1つの区間を選ぶ（1~30区間すべてから）
    noise_interval = random.randint(0, NUM_INTERVALS - 1)
    
    # ノイズを付与
    noisy_psd, start_idx, end_idx = add_noise_to_interval(
        original_psd,
        noise_interval,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL,
        num_intervals=NUM_INTERVALS
    )
    
    noisy_data.append(noisy_psd)
    labels.append(noise_interval)
    
    if (i + 1) % 1000 == 0:
        print(f"  処理済み: {i+1:,} / {num_samples:,}")

# Tensorに変換
noisy_data = torch.stack(noisy_data)  # (32000, 3000)
labels = torch.tensor(labels, dtype=torch.long)  # (32000,)

print(f"\nデータセットの形状:")
print(f"  入力データ: {noisy_data.shape}")
print(f"  ラベル: {labels.shape}")

# データの分割（訓練:80%, 検証:10%, テスト:10%）
print("\nデータを分割中...")
indices = list(range(num_samples))
random.shuffle(indices)

train_size = int(num_samples * 0.8)
val_size = int(num_samples * 0.1)
test_size = num_samples - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_data = noisy_data[train_indices]
train_labels = labels[train_indices]
val_data = noisy_data[val_indices]
val_labels = labels[val_indices]
test_data = noisy_data[test_indices]
test_labels = labels[test_indices]

print(f"訓練データ: {len(train_indices):,}サンプル")
print(f"検証データ: {len(val_indices):,}サンプル")
print(f"テストデータ: {len(test_indices):,}サンプル")

# ラベルの分布を確認
print("\nラベルの分布（訓練データ）:")
label_counts = torch.bincount(train_labels)
for i in range(NUM_INTERVALS):
    print(f"  区間 {i+1:2d}: {label_counts[i].item():4d}サンプル")

# データセットを保存
print("\nデータセットを保存中...")
dataset = {
    'train': {
        'data': train_data,
        'labels': train_labels
    },
    'val': {
        'data': val_data,
        'labels': val_labels
    },
    'test': {
        'data': test_data,
        'labels': test_labels
    },
    'config': {
        'num_intervals': NUM_INTERVALS,
        'noise_type': NOISE_TYPE,
        'noise_level': NOISE_LEVEL
    }
}

with open('baseline_dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)

print("データセットを 'baseline_dataset.pickle' に保存しました")
print("\nデータセット準備完了！")

