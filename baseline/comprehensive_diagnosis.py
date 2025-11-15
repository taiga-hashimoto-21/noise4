"""
学習が進まない原因を包括的に診断するスクリプト
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# モデルをインポート
import sys
sys.path.append('.')
from baseline_model import SimpleCNN

# データセットクラス
class PSDDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

print("=" * 60)
print("包括的な診断: 学習が進まない原因の追求")
print("=" * 60)

# 1. データの読み込みと確認
print("\n【ステップ1】データの確認")
print("-" * 60)
with open('baseline_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)

train_data = dataset['train']['data']
train_labels = dataset['train']['labels']

if not isinstance(train_data, torch.Tensor):
    train_data = torch.FloatTensor(train_data)
if not isinstance(train_labels, torch.Tensor):
    train_labels = torch.LongTensor(train_labels)

print(f"✓ データ形状: {train_data.shape}")
print(f"✓ ラベル形状: {train_labels.shape}")
print(f"✓ データ範囲: [{train_data.min():.6e}, {train_data.max():.6e}]")
print(f"✓ データ平均: {train_data.mean():.6e}")
print(f"✓ データ標準偏差: {train_data.std():.6e}")

# 2. モデルの初期化と確認
print("\n【ステップ2】モデルの初期化確認")
print("-" * 60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")

model = SimpleCNN(num_classes=10).to(device)
print(f"✓ モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# モデルの初期出力を確認
model.eval()
with torch.no_grad():
    sample_input = train_data[:1].to(device)
    initial_output = model(sample_input)
    initial_probs = torch.softmax(initial_output, dim=1)
    
    print(f"\n初期出力:")
    print(f"  出力値: {initial_output[0].tolist()}")
    print(f"  確率分布: {initial_probs[0].tolist()}")
    print(f"  予測クラス: {initial_output[0].argmax().item()}")
    print(f"  出力の範囲: [{initial_output.min():.4f}, {initial_output.max():.4f}]")
    print(f"  出力の平均: {initial_output.mean():.4f}")
    print(f"  出力の標準偏差: {initial_output.std():.4f}")

# 3. 損失関数の確認
print("\n【ステップ3】損失関数の確認")
print("-" * 60)
criterion = nn.CrossEntropyLoss()

# サンプルデータで損失を計算
sample_data = train_data[:10].to(device)
sample_labels = train_labels[:10].to(device)

model.eval()
with torch.no_grad():
    sample_output = model(sample_data)
    sample_loss = criterion(sample_output, sample_labels)
    
    print(f"✓ サンプル損失: {sample_loss.item():.4f}")
    print(f"✓ 理論的な最大損失（ランダム）: {np.log(10):.4f}")
    print(f"✓ 理論的な最小損失（完璧）: 0.0")
    
    if sample_loss.item() > 2.0:
        print("⚠ 警告: 損失が高いです（ランダム推測レベル）")
    elif sample_loss.item() < 0.1:
        print("✓ 損失は低いです（良い状態）")
    else:
        print("→ 損失は中程度です")

# 4. 勾配の確認
print("\n【ステップ4】勾配の確認")
print("-" * 60)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 1ステップの学習を実行
optimizer.zero_grad()
output = model(sample_data)
loss = criterion(output, sample_labels)
loss.backward()

# 勾配を確認
total_grad_norm = 0
zero_grad_count = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm ** 2
        if grad_norm < 1e-8:
            zero_grad_count += 1
            print(f"⚠ {name}: 勾配がほぼゼロ ({grad_norm:.2e})")
    else:
        zero_grad_count += 1
        print(f"⚠ {name}: 勾配がNone")

total_grad_norm = np.sqrt(total_grad_norm)
print(f"\n✓ 総勾配ノルム: {total_grad_norm:.6f}")
print(f"✓ 勾配がゼロのパラメータ数: {zero_grad_count}")

if total_grad_norm < 1e-6:
    print("⚠ 警告: 勾配が極端に小さいです（勾配消失の可能性）")
elif total_grad_norm > 100:
    print("⚠ 警告: 勾配が極端に大きいです（勾配爆発の可能性）")
else:
    print("✓ 勾配は適切な範囲内です")

# 5. 重みの更新確認
print("\n【ステップ5】重みの更新確認")
print("-" * 60)
# 最初の層の重みを保存
first_layer_weight_before = model.model.conv1.weight.data.clone()

optimizer.step()

first_layer_weight_after = model.model.conv1.weight.data.clone()
weight_change = (first_layer_weight_after - first_layer_weight_before).abs().mean().item()

print(f"✓ 重みの変化（平均）: {weight_change:.8f}")
print(f"✓ 重みの変化率: {weight_change / first_layer_weight_before.abs().mean().item() * 100:.4f}%")

if weight_change < 1e-8:
    print("⚠ 警告: 重みが更新されていません")
else:
    print("✓ 重みは更新されています")

# 6. データの前処理の影響確認
print("\n【ステップ6】データの前処理の影響確認")
print("-" * 60)

# ログ変換を適用
log_data = torch.log(train_data.clamp(min=1e-30))
print(f"ログ変換後:")
print(f"  平均: {log_data.mean():.4f}")
print(f"  標準偏差: {log_data.std():.4f}")
print(f"  範囲: [{log_data.min():.4f}, {log_data.max():.4f}]")

# スケーリングを適用
scaled_data = train_data * 1e20
print(f"\nスケーリング後（×1e20）:")
print(f"  平均: {scaled_data.mean():.4f}")
print(f"  標準偏差: {scaled_data.std():.4f}")
print(f"  範囲: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")

# 7. 学習のシミュレーション
print("\n【ステップ7】学習のシミュレーション（5ステップ）")
print("-" * 60)

# モデルを再初期化
model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_dataset = PSDDataset(train_data[:100], train_labels[:100])  # 小さなデータセット
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.train()
losses = []
for step, (data, labels) in enumerate(train_loader):
    if step >= 5:
        break
    
    data = data.to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    
    # 勾配ノルムを計算
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e10)
    
    optimizer.step()
    
    losses.append(loss.item())
    print(f"  ステップ {step+1}: 損失={loss.item():.4f}, 勾配ノルム={grad_norm:.6f}")

if len(losses) > 1:
    loss_change = losses[-1] - losses[0]
    print(f"\n✓ 損失の変化: {loss_change:.4f}")
    if loss_change > 0:
        print("⚠ 警告: 損失が増加しています")
    elif abs(loss_change) < 0.01:
        print("⚠ 警告: 損失がほとんど変化していません")
    else:
        print("✓ 損失は減少しています")

# 8. まとめと推奨事項
print("\n【ステップ8】診断結果のまとめ")
print("=" * 60)

issues = []
recommendations = []

# データのスケールチェック
if train_data.std() < 1e-20:
    issues.append("データのスケールが極端に小さい")
    recommendations.append("ログ変換を適用: x = torch.log(x.clamp(min=1e-30))")

# 勾配チェック
if total_grad_norm < 1e-6:
    issues.append("勾配が極端に小さい（勾配消失）")
    recommendations.append("学習率を上げる、またはバッチ正規化を調整")

if total_grad_norm > 100:
    issues.append("勾配が極端に大きい（勾配爆発）")
    recommendations.append("勾配クリッピングを適用、または学習率を下げる")

# 重み更新チェック
if weight_change < 1e-8:
    issues.append("重みが更新されていない")
    recommendations.append("学習率を確認、またはオプティマイザを確認")

# 損失チェック
if sample_loss.item() > 2.0:
    issues.append("損失が高い（ランダム推測レベル）")
    recommendations.append("モデルの初期化を確認、またはデータの前処理を確認")

if issues:
    print("発見された問題:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n推奨される対策:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("✓ 明らかな問題は見つかりませんでした")
    print("  より詳細なログを追加して、学習過程を監視することを推奨します")

print("\n" + "=" * 60)

