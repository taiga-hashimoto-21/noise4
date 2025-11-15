"""
学習が進まない原因を診断するスクリプト
"""

import pickle
import torch
import numpy as np

# データセットの読み込み
print("データセットを読み込み中...")
with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['train']['data']
train_labels = dataset['train']['labels']

# Tensorに変換
if not isinstance(train_data, torch.Tensor):
    train_data = torch.FloatTensor(train_data)
if not isinstance(train_labels, torch.Tensor):
    train_labels = torch.LongTensor(train_labels)

print(f"\n=== データの統計情報 ===")
print(f"データの形状: {train_data.shape}")
print(f"データの型: {train_data.dtype}")
print(f"データの平均: {train_data.mean():.6e}")
print(f"データの標準偏差: {train_data.std():.6e}")
print(f"データの最小値: {train_data.min():.6e}")
print(f"データの最大値: {train_data.max():.6e}")
print(f"データにNaNがあるか: {torch.isnan(train_data).any()}")
print(f"データにInfがあるか: {torch.isinf(train_data).any()}")

# サンプルを確認
print(f"\n=== サンプルデータの確認 ===")
sample = train_data[0]
print(f"サンプル1の平均: {sample.mean():.6e}")
print(f"サンプル1の標準偏差: {sample.std():.6e}")
print(f"サンプル1の最小値: {sample.min():.6e}")
print(f"サンプル1の最大値: {sample.max():.6e}")

# ノイズ区間と他の区間の差を確認
print(f"\n=== ノイズ区間と他の区間の差（サンプル1） ===")
label = train_labels[0].item()
start_idx = label * 300
end_idx = start_idx + 300

noise_region = sample[start_idx:end_idx]
other_regions = torch.cat([sample[:start_idx], sample[end_idx:]])

print(f"ラベル: {label}")
print(f"ノイズ区間: {start_idx}-{end_idx}")
print(f"ノイズ区間の平均: {noise_region.mean():.6e}")
print(f"ノイズ区間の標準偏差: {noise_region.std():.6e}")
print(f"他の区間の平均: {other_regions.mean():.6e}")
print(f"他の区間の標準偏差: {other_regions.std():.6e}")
print(f"差: {abs(noise_region.mean() - other_regions.mean()):.6e}")
if other_regions.mean() != 0:
    ratio = noise_region.mean() / other_regions.mean()
    print(f"比: {ratio:.2f}x")

# データのスケールが適切か確認
print(f"\n=== データのスケール評価 ===")
data_scale = train_data.std()
print(f"データの標準偏差: {data_scale:.6e}")

if data_scale < 1e-20:
    print("⚠ 警告: データのスケールが極端に小さいです")
    print("  これにより、モデルの学習が困難になる可能性があります")
    print("  推奨: ログ変換やスケーリングを検討してください")
elif data_scale < 1e-10:
    print("⚠ 警告: データのスケールが非常に小さいです")
    print("  推奨: スケーリングを検討してください")
elif data_scale > 1e10:
    print("⚠ 警告: データのスケールが非常に大きいです")
    print("  推奨: 正規化を検討してください")
else:
    print("✓ データのスケールは適切です")

# ノイズの検出可能性を評価
print(f"\n=== ノイズの検出可能性評価 ===")
all_ratios = []
for i in range(min(100, len(train_data))):
    sample = train_data[i]
    label = train_labels[i].item()
    start_idx = label * 300
    end_idx = start_idx + 300
    
    noise_region = sample[start_idx:end_idx]
    other_regions = torch.cat([sample[:start_idx], sample[end_idx:]])
    
    if other_regions.mean() != 0:
        ratio = abs(noise_region.mean() / other_regions.mean())
        all_ratios.append(ratio)

if all_ratios:
    avg_ratio = np.mean(all_ratios)
    print(f"平均的なノイズ/他区間比: {avg_ratio:.2f}x")
    if avg_ratio < 1.5:
        print("⚠ 警告: ノイズと他の区間の差が小さいです")
        print("  これにより、モデルがノイズを検出するのが困難です")
    elif avg_ratio < 3.0:
        print("⚠ 注意: ノイズと他の区間の差は中程度です")
        print("  モデルが学習できる可能性がありますが、難しいかもしれません")
    else:
        print("✓ ノイズと他の区間の差は十分に大きいです")

print("\n=== 推奨される対策 ===")
print("1. データのスケールが極端に小さい場合:")
print("   - ログ変換を適用: x = torch.log(x.clamp(min=1e-30))")
print("   - または、スケーリング: x = x * 1e20")
print("2. ノイズの検出可能性が低い場合:")
print("   - ノイズレベルをさらに上げる（NOISE_LEVEL > 5.0）")
print("   - または、異なるノイズタイプを試す")
print("3. モデルの学習が進まない場合:")
print("   - 学習率を調整する（0.001 → 0.01 → 0.1）")
print("   - バッチサイズを小さくする（64 → 32 → 16）")
print("   - よりシンプルなモデルを試す")

