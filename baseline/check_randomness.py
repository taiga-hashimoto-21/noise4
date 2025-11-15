"""
ノイズがランダムに付与されているかを確認するスクリプト
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# データセットの読み込み
print("データセットを読み込み中...")
with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

train_labels = dataset['train']['labels']
val_labels = dataset['val']['labels']
test_labels = dataset['test']['labels']

# ラベルをnumpy配列に変換
if hasattr(train_labels, 'numpy'):
    train_labels = train_labels.numpy()
elif hasattr(train_labels, 'cpu'):
    train_labels = train_labels.cpu().numpy()
else:
    train_labels = np.array(train_labels)

if hasattr(val_labels, 'numpy'):
    val_labels = val_labels.numpy()
elif hasattr(val_labels, 'cpu'):
    val_labels = val_labels.cpu().numpy()
else:
    val_labels = np.array(val_labels)

if hasattr(test_labels, 'numpy'):
    test_labels = test_labels.numpy()
elif hasattr(test_labels, 'cpu'):
    test_labels = test_labels.cpu().numpy()
else:
    test_labels = np.array(test_labels)

# ラベルの分布を確認
print("\n=== ラベルの分布（訓練データ） ===")
train_label_counts = Counter(train_labels)
for i in range(10):
    count = train_label_counts.get(i, 0)
    percentage = count / len(train_labels) * 100
    print(f"  クラス {i}: {count:4d}サンプル ({percentage:5.2f}%)")

print("\n=== ラベルの分布（検証データ） ===")
val_label_counts = Counter(val_labels)
for i in range(10):
    count = val_label_counts.get(i, 0)
    percentage = count / len(val_labels) * 100
    print(f"  クラス {i}: {count:4d}サンプル ({percentage:5.2f}%)")

print("\n=== ラベルの分布（テストデータ） ===")
test_label_counts = Counter(test_labels)
for i in range(10):
    count = test_label_counts.get(i, 0)
    percentage = count / len(test_labels) * 100
    print(f"  クラス {i}: {count:4d}サンプル ({percentage:5.2f}%)")

# 期待値（均等分布の場合）
expected_per_class = len(train_labels) / 10
print(f"\n期待値（均等分布の場合）: {expected_per_class:.1f}サンプル/クラス")

# カイ二乗検定でランダム性を確認
from scipy.stats import chisquare

observed = [train_label_counts.get(i, 0) for i in range(10)]
expected = [expected_per_class] * 10
chi2_stat, p_value = chisquare(observed, expected)

print(f"\n=== カイ二乗検定（ランダム性の検証） ===")
print(f"カイ二乗統計量: {chi2_stat:.4f}")
print(f"p値: {p_value:.6f}")
if p_value > 0.05:
    print("✓ ランダムに分布している（p > 0.05）")
else:
    print("⚠ ランダムではない可能性がある（p <= 0.05）")

# 連続するラベルのパターンを確認（ランダムなら連続パターンは少ない）
print("\n=== 連続するラベルのパターン確認 ===")
consecutive_same = 0
for i in range(len(train_labels) - 1):
    if train_labels[i] == train_labels[i + 1]:
        consecutive_same += 1

consecutive_ratio = consecutive_same / (len(train_labels) - 1)
print(f"連続して同じラベル: {consecutive_same}回 ({consecutive_ratio*100:.2f}%)")
if consecutive_ratio < 0.15:  # ランダムなら約10%程度
    print("✓ 連続パターンが少なく、ランダムに分布している")
else:
    print("⚠ 連続パターンが多く、ランダムではない可能性がある")

# 可視化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 訓練データのラベル分布
axes[0].bar(range(10), [train_label_counts.get(i, 0) for i in range(10)])
axes[0].axhline(y=expected_per_class, color='r', linestyle='--', label=f'期待値 ({expected_per_class:.0f})')
axes[0].set_xlabel('クラス')
axes[0].set_ylabel('サンプル数')
axes[0].set_title('訓練データのラベル分布')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 検証データのラベル分布
val_expected = len(val_labels) / 10
axes[1].bar(range(10), [val_label_counts.get(i, 0) for i in range(10)])
axes[1].axhline(y=val_expected, color='r', linestyle='--', label=f'期待値 ({val_expected:.0f})')
axes[1].set_xlabel('クラス')
axes[1].set_ylabel('サンプル数')
axes[1].set_title('検証データのラベル分布')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# テストデータのラベル分布
test_expected = len(test_labels) / 10
axes[2].bar(range(10), [test_label_counts.get(i, 0) for i in range(10)])
axes[2].axhline(y=test_expected, color='r', linestyle='--', label=f'期待値 ({test_expected:.0f})')
axes[2].set_xlabel('クラス')
axes[2].set_ylabel('サンプル数')
axes[2].set_title('テストデータのラベル分布')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('label_distribution_check.png', dpi=150, bbox_inches='tight')
print("\n✓ ラベル分布を 'label_distribution_check.png' に保存しました")

plt.show()

