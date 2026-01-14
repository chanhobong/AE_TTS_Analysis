# Flow Matching & DDPM Models

## Overview

Flow Matching과 DDPM 모델을 사용한 Latent Space Anomaly Detection 구현.

## 파일 구조

- `latent_velocity_estimator.py`: 3D Convolution 기반 ResNet 모델 (Time Conditioning)
- `flow_matching.py`: Flow Matching 학습 및 Anomaly Detection 함수
- `ddpm.py`: DDPM 학습 및 Anomaly Detection 함수
- `__init__.py`: 패키지 초기화

## 모델 구조

### LatentVelocityEstimator

- **입력**: `z_t` (B, C, D, H, W) latent, `t` (B,) time step
- **출력**: `v_θ(z_t, t)` (B, C, D, H, W) velocity field
- **구조**: 3D ResNet with Time Embedding
- **파라미터**: 약 1.38M (base_channels=64 기준)

### Flow Matching

- **Forward Path**: `z_t = (1-t)z_0 + t·z_1`
- **Target Velocity**: `u_t = z_1 - z_0`
- **Loss**: `MSE(v_θ(z_t, t), u_t)`

## Anomaly Detection

### 1. Velocity Inconsistency (추천)

```python
from models import LatentVelocityEstimator, compute_anomaly_score_velocity_inconsistency

scores = compute_anomaly_score_velocity_inconsistency(
    model, z_test, n_samples=10
)
```

여러 `t` 구간에서 예측된 velocity의 variance를 계산.

### 2. MSE Multi-t

```python
from models import compute_anomaly_score_mse_multi_t

scores = compute_anomaly_score_mse_multi_t(
    model, z_test, n_samples=10
)
```

여러 `t` 구간에서의 MSE 오차를 평균.

### 3. Round-trip Error (옵션)

```python
from models import compute_anomaly_score_round_trip

scores = compute_anomaly_score_round_trip(
    model, z_test, n_steps=50
)
```

ODE Solver를 사용한 round-trip error.

## 사용 예제

### Flow Matching 학습

```bash
python utils/train_flow_matching.py \
  --ae_model_path outputs/train_ae_normal_only/model.pth \
  --normal_root /path/to/normal_RM_V2 \
  --metadata_csv /path/to/Combined_Labels.csv \
  --train_csv data/train.csv \
  --base_channels 64 \
  --epochs 100 \
  --batch_size 4 \
  --out_dir outputs/flow_matching
```

### DDPM 학습

```bash
python utils/train_ddpm.py \
  --ae_model_path outputs/train_ae_normal_only/model.pth \
  --normal_root /path/to/normal_RM_V2 \
  --metadata_csv /path/to/Combined_Labels.csv \
  --train_csv data/train.csv \
  --base_channels 64 \
  --timesteps 1000 \
  --schedule_type cosine \
  --epochs 100 \
  --batch_size 4 \
  --out_dir outputs/ddpm
```

### 모델 테스트

#### Flow Matching

```python
import torch
from models import LatentVelocityEstimator

device = torch.device("mps")
model = LatentVelocityEstimator(
    in_channels=128,
    base_channels=64,
).to(device)

z_t = torch.randn(2, 128, 8, 16, 16).to(device)
t = torch.rand(2).to(device)

v_t = model(z_t, t)
print(f"Output shape: {v_t.shape}")
```

#### DDPM

```python
import torch
from models import LatentVelocityEstimator, NoiseSchedule, compute_anomaly_score_ddpm

device = torch.device("mps")
model = LatentVelocityEstimator(
    in_channels=128,
    base_channels=64,
).to(device)

noise_schedule = NoiseSchedule(timesteps=1000, schedule_type="cosine")
z_test = torch.randn(2, 128, 8, 16, 16).to(device)

scores = compute_anomaly_score_ddpm(model, z_test, noise_schedule, device=device)
print(f"Anomaly scores: {scores}")
```

## 하드웨어 요구사항

- **M5 MacBook**: MPS (Metal Performance Shaders) 사용
- **권장 설정**: base_channels=64, batch_size=4
- **메모리**: 약 2-4GB VRAM

## 참고

- 입력 latent shape: (Batch, 128, 8, 16, 16)
- Time embedding dimension: 128 (default)
- Residual blocks: 4 (default)
