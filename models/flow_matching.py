"""
flow_matching.py
- Conditional Flow Matching (CFM) 학습 로직
- Anomaly Detection 함수
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def sample_flow_matching_pair(z_0: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flow Matching을 위한 (z_0, z_1) 페어 생성.
    
    Args:
        z_0: (B, C, D, H, W) Normal latent samples
        device: Device
    Returns:
        z_1: (B, C, D, H, W) Gaussian noise (N(0, I))
        z_t: (B, C, D, H, W) Interpolated latent at time t
        t: (B,) time steps
        u_t: (B, C, D, H, W) target velocity
    """
    batch_size = z_0.shape[0]
    
    # z_1: Standard Gaussian noise
    z_1 = torch.randn_like(z_0)
    
    # t: Uniform sampling from [0, 1]
    t = torch.rand(batch_size, device=device)  # (B,)
    t_expanded = t[:, None, None, None, None]  # (B, 1, 1, 1, 1) for broadcasting
    
    # Linear interpolation: z_t = (1-t)z_0 + t*z_1
    z_t = (1 - t_expanded) * z_0 + t_expanded * z_1
    
    # Target velocity: u_t = z_1 - z_0
    u_t = z_1 - z_0
    
    # Return t as (B,) - keep original shape, don't squeeze (squeeze() removes all dims of size 1)
    # t is already (B,), so we just return it
    
    return z_t, t, u_t


def compute_flow_matching_loss(
    model: nn.Module,
    z_0: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """
    Flow Matching Loss 계산.
    
    Args:
        model: LatentVelocityEstimator
        z_0: (B, C, D, H, W) Normal latent samples
        device: Device
    Returns:
        loss: Scalar loss
        info: Dictionary with additional info
    """
    # Sample flow matching pair
    z_t, t, u_t = sample_flow_matching_pair(z_0, device)
    
    # Predict velocity
    v_t = model(z_t, t)  # (B, C, D, H, W)
    
    # MSE Loss
    loss = nn.functional.mse_loss(v_t, u_t)
    
    info = {
        "z_t": z_t.detach(),
        "t": t.detach(),
        "u_t": u_t.detach(),
        "v_t": v_t.detach(),
    }
    
    return loss, info


@torch.no_grad()
def compute_anomaly_score_velocity_inconsistency(
    model: nn.Module,
    z_test: torch.Tensor,
    n_samples: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Velocity Inconsistency 기반 Anomaly Score 계산 (보조/추가 분석용).
    
    여러 t 구간에서 예측된 velocity의 variance를 계산.
    
    Note: 이 메트릭은 정상이어도 분산이 클 수 있고, t/z1 샘플링 랜덤성을 반영하기 쉽습니다.
    기본 anomaly score로는 compute_anomaly_score_mse_multi_t를 권장합니다.
    
    Args:
        model: Trained LatentVelocityEstimator
        z_test: (B, C, D, H, W) or (C, D, H, W) test latent
        n_samples: Number of t samples for variance calculation
        device: Device
    Returns:
        scores: (B,) anomaly scores (higher = more anomalous)
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Add batch dimension if needed
    if z_test.dim() == 4:
        z_test = z_test.unsqueeze(0)
    batch_size = z_test.shape[0]
    
    # Move to device
    z_test = z_test.to(device)
    
    # Sample multiple time steps
    t_samples = torch.rand(n_samples, batch_size, device=device)  # (n_samples, B)
    
    # Sample z_1 for each (sample, batch) pair: (n_samples, B, C, D, H, W)
    z_1_samples = torch.randn(n_samples, batch_size, *z_test.shape[1:], device=device)
    
    # Compute velocities for each (t, z_1) pair
    velocities = []
    for i in range(n_samples):
        t_i = t_samples[i]  # (B,)
        z_1_i = z_1_samples[i]  # (B, C, D, H, W)
        
        # Interpolate: z_t = (1-t)z_test + t*z_1
        t_i_expanded = t_i[:, None, None, None, None]  # (B, 1, 1, 1, 1)
        z_t_i = (1 - t_i_expanded) * z_test + t_i_expanded * z_1_i
        
        # Predict velocity
        v_t_i = model(z_t_i, t_i)  # (B, C, D, H, W)
        velocities.append(v_t_i)
    
    # Stack: (n_samples, B, C, D, H, W)
    velocities = torch.stack(velocities, dim=0)
    
    # Compute variance across samples (for each test sample)
    # Variance over n_samples dimension
    velocity_var = torch.var(velocities, dim=0)  # (B, C, D, H, W)
    
    # Aggregate: mean over spatial dimensions
    scores = velocity_var.mean(dim=(1, 2, 3, 4))  # (B,)
    
    return scores


@torch.no_grad()
def compute_anomaly_score_mse_multi_t(
    model: nn.Module,
    z_test: torch.Tensor,
    n_samples: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    여러 t 구간에서 발생하는 MSE 오차를 Anomaly Score로 계산 (추천 메트릭).
    
    수식: S_FM(z) = E_{t,z_1}[||v_θ(z_t, t) - (z_1 - z)||²]
    
    이는 Flow Matching의 기본 anomaly score로, 예측된 velocity와 target velocity 간의 
    오차를 여러 (t, z_1) 샘플에 대해 평균한 값입니다.
    
    Args:
        model: Trained LatentVelocityEstimator
        z_test: (B, C, D, H, W) or (C, D, H, W) test latent
        n_samples: Number of (t, z_1) samples to average over
        device: Device
    Returns:
        scores: (B,) anomaly scores (higher = more anomalous)
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Add batch dimension if needed
    if z_test.dim() == 4:
        z_test = z_test.unsqueeze(0)
    batch_size = z_test.shape[0]
    
    # Move to device
    z_test = z_test.to(device)
    
    # Sample multiple time steps
    t_samples = torch.rand(n_samples, batch_size, device=device)  # (n_samples, B)
    
    # Sample z_1 for each (sample, batch) pair: (n_samples, B, C, D, H, W)
    z_1_samples = torch.randn(n_samples, batch_size, *z_test.shape[1:], device=device)
    
    # Compute MSE errors
    mse_errors = []
    for i in range(n_samples):
        t_i = t_samples[i]  # (B,)
        z_1_i = z_1_samples[i]  # (B, C, D, H, W)
        
        # Interpolate: z_t = (1-t)z_test + t*z_1
        t_i_expanded = t_i[:, None, None, None, None]  # (B, 1, 1, 1, 1)
        z_t_i = (1 - t_i_expanded) * z_test + t_i_expanded * z_1_i
        
        # Predict velocity
        v_t_i = model(z_t_i, t_i)  # (B, C, D, H, W)
        
        # Target velocity: u_t = z_1 - z_test
        u_t_i = z_1_i - z_test  # (B, C, D, H, W)
        
        # MSE error
        mse_i = nn.functional.mse_loss(v_t_i, u_t_i, reduction='none')  # (B, C, D, H, W)
        mse_i = mse_i.mean(dim=(1, 2, 3, 4))  # (B,)
        mse_errors.append(mse_i)
    
    # Stack: (n_samples, B)
    mse_errors = torch.stack(mse_errors, dim=0)
    
    # Aggregate: mean over n_samples
    scores = mse_errors.mean(dim=0)  # (B,)
    
    return scores


def euler_ode_solver(
    model: nn.Module,
    z_0: torch.Tensor,
    n_steps: int = 50,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Euler method를 사용한 ODE Solver (forward: z_0 -> z_1).
    
    dz/dt = v_θ(z_t, t)를 수치적으로 풂.
    
    Args:
        model: Trained LatentVelocityEstimator
        z_0: (B, C, D, H, W) initial latent
        n_steps: Number of Euler steps
        device: Device
    Returns:
        z_1: (B, C, D, H, W) final latent
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    z_t = z_0.to(device)
    dt = 1.0 / n_steps
    
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((z_t.shape[0],), i * dt, device=device)
            v_t = model(z_t, t)
            z_t = z_t + dt * v_t
    
    return z_t


@torch.no_grad()
def compute_anomaly_score_round_trip(
    model: nn.Module,
    z_test: torch.Tensor,
    n_steps: int = 50,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Round-trip Error 기반 Anomaly Score (옵션).
    
    z_test -> ODE -> z_1 -> 역ODE -> z_0_recon
    Score = ||z_test - z_0_recon||
    
    Note: 역 ODE는 간단히 구현 (실제로는 더 복잡할 수 있음)
    
    Args:
        model: Trained LatentVelocityEstimator
        z_test: (B, C, D, H, W) or (C, D, H, W) test latent
        n_steps: Number of Euler steps
        device: Device
    Returns:
        scores: (B,) anomaly scores (higher = more anomalous)
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Add batch dimension if needed
    if z_test.dim() == 4:
        z_test = z_test.unsqueeze(0)
    
    z_test = z_test.to(device)
    
    # Forward ODE: z_test -> z_1
    z_1 = euler_ode_solver(model, z_test, n_steps=n_steps, device=device)
    
    # Reverse ODE: z_1 -> z_0_recon
    # 역 ODE: dz/dt = -v_θ(z_t, 1-t)
    z_recon = z_1
    dt = 1.0 / n_steps
    
    with torch.no_grad():
        for i in range(n_steps):
            t_reverse = 1.0 - (i * dt)  # Reverse time
            t = torch.full((z_recon.shape[0],), t_reverse, device=device)
            v_t = model(z_recon, t)
            z_recon = z_recon - dt * v_t  # Reverse direction
    
    # Round-trip error
    error = nn.functional.mse_loss(z_test, z_recon, reduction='none')  # (B, C, D, H, W)
    scores = error.mean(dim=(1, 2, 3, 4))  # (B,)
    
    return scores


if __name__ == "__main__":
    # 간단한 테스트
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from latent_velocity_estimator import LatentVelocityEstimator
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 생성
    model = LatentVelocityEstimator(
        in_channels=128,
        base_channels=64,
    ).to(device)
    
    # Test data
    batch_size = 2
    z_0 = torch.randn(batch_size, 128, 8, 16, 16).to(device)
    
    # Test flow matching loss
    print("\n--- Testing Flow Matching Loss ---")
    loss, info = compute_flow_matching_loss(model, z_0, device)
    print(f"Loss: {loss.item():.4f}")
    print(f"z_t shape: {info['z_t'].shape}")
    print(f"t shape: {info['t'].shape}")
    
    # Test anomaly score (velocity inconsistency)
    print("\n--- Testing Anomaly Score (Velocity Inconsistency) ---")
    z_test = torch.randn(batch_size, 128, 8, 16, 16).to(device)
    scores = compute_anomaly_score_velocity_inconsistency(model, z_test, n_samples=5, device=device)
    print(f"Anomaly scores: {scores}")
    
    # Test anomaly score (MSE multi-t)
    print("\n--- Testing Anomaly Score (MSE Multi-t) ---")
    scores_mse = compute_anomaly_score_mse_multi_t(model, z_test, n_samples=5, device=device)
    print(f"Anomaly scores (MSE): {scores_mse}")
    
    print("\n✅ All tests passed!")
