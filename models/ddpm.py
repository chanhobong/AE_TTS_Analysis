"""
ddpm.py
- Denoising Diffusion Probabilistic Model (DDPM) 학습 로직
- Latent space에서의 DDPM
- Anomaly Detection 함수
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear noise schedule for DDPM.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level
    Returns:
        betas: (timesteps,) noise schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule for DDPM (often better than linear).
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset for numerical stability
    Returns:
        betas: (timesteps,) noise schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class NoiseSchedule:
    """
    DDPM noise schedule manager.
    Precomputes α_t, √α_t, √(1-α_t), etc.
    """
    def __init__(self, timesteps: int = 1000, schedule_type: str = "cosine", device: torch.device = None):
        """
        Args:
            timesteps: Number of diffusion steps
            schedule_type: "linear" or "cosine"
            device: Device to store tensors on (if None, uses CPU)
        """
        self.timesteps = timesteps
        self.device = device if device is not None else torch.device("cpu")
        self._buffers = {}
        
        if schedule_type == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Move to device
        betas = betas.to(self.device)
        
        # Convert to tensors
        self._buffers["betas"] = betas
        
        # Compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]])
        
        # Precompute values
        self._buffers["alphas"] = alphas
        self._buffers["alphas_cumprod"] = alphas_cumprod
        self._buffers["alphas_cumprod_prev"] = alphas_cumprod_prev
        self._buffers["sqrt_alphas_cumprod"] = torch.sqrt(alphas_cumprod)
        self._buffers["sqrt_one_minus_alphas_cumprod"] = torch.sqrt(1.0 - alphas_cumprod)
    
    def __getattr__(self, name: str):
        if '_buffers' in self.__dict__ and name in self._buffers:
            return self._buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def sample_ddpm_pair(
    z_0: torch.Tensor,
    noise_schedule: NoiseSchedule,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DDPM forward diffusion: Sample (z_0, z_t, ε, t).
    
    Args:
        z_0: (B, C, D, H, W) Normal latent samples
        noise_schedule: NoiseSchedule object
        device: Device
    Returns:
        z_t: (B, C, D, H, W) Noisy latent at time t
        epsilon: (B, C, D, H, W) Noise that was added
        t: (B,) Random timesteps
    """
    batch_size = z_0.shape[0]
    
    # Sample random timesteps
    t = torch.randint(0, noise_schedule.timesteps, (batch_size,), device=device).long()
    
    # Sample noise
    epsilon = torch.randn_like(z_0)
    
    # Get noise schedule values for each sample in batch
    sqrt_alphas_cumprod_t = noise_schedule.sqrt_alphas_cumprod[t]  # (B,)
    sqrt_one_minus_alphas_cumprod_t = noise_schedule.sqrt_one_minus_alphas_cumprod[t]  # (B,)
    
    # Expand for broadcasting: (B, 1, 1, 1, 1)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None, None]
    
    # Forward diffusion: z_t = √(α_t) z_0 + √(1-α_t) ε
    z_t = sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * epsilon
    
    # Normalize t to [0, 1] for model input
    t_normalized = t.float() / noise_schedule.timesteps
    
    return z_t, epsilon, t_normalized


def compute_ddpm_loss(
    model: nn.Module,
    z_0: torch.Tensor,
    noise_schedule: NoiseSchedule,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    """
    DDPM Loss 계산.
    
    Args:
        model: LatentVelocityEstimator (reused as noise predictor)
        z_0: (B, C, D, H, W) Normal latent samples
        noise_schedule: NoiseSchedule object
        device: Device
    Returns:
        loss: Scalar loss
        info: Dictionary with additional info
    """
    # Sample forward diffusion pair
    z_t, epsilon, t = sample_ddpm_pair(z_0, noise_schedule, device)
    
    # Predict noise: model outputs velocity, but we interpret it as noise
    # (Note: For DDPM, we could use the same architecture but interpret output differently)
    epsilon_pred = model(z_t, t)  # (B, C, D, H, W)
    
    # MSE Loss
    loss = nn.functional.mse_loss(epsilon_pred, epsilon)
    
    info = {
        "z_t": z_t.detach(),
        "epsilon": epsilon.detach(),
        "epsilon_pred": epsilon_pred.detach(),
        "t": t.detach(),
    }
    
    return loss, info


@torch.no_grad()
def compute_anomaly_score_ddpm(
    model: nn.Module,
    z_test: torch.Tensor,
    noise_schedule: NoiseSchedule,
    n_samples: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    DDPM 기반 Anomaly Score 계산.
    
    여러 t 구간에서 예측된 noise와 실제 noise의 오차를 계산.
    
    Args:
        model: Trained DDPM model (noise predictor)
        z_test: (B, C, D, H, W) or (C, D, H, W) test latent
        noise_schedule: NoiseSchedule object
        n_samples: Number of timesteps to sample
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
    
    z_test = z_test.to(device)
    
    # Sample multiple timesteps
    errors = []
    for _ in range(n_samples):
        # Sample random timestep
        t = torch.randint(0, noise_schedule.timesteps, (batch_size,), device=device).long()
        t_normalized = t.float() / noise_schedule.timesteps
        
        # Sample noise
        epsilon = torch.randn_like(z_test)
        
        # Get noise schedule values
        sqrt_alphas_cumprod_t = noise_schedule.sqrt_alphas_cumprod[t]  # (B,)
        sqrt_one_minus_alphas_cumprod_t = noise_schedule.sqrt_one_minus_alphas_cumprod[t]  # (B,)
        
        # Expand for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None, None]
        
        # Forward diffusion
        z_t = sqrt_alphas_cumprod_t * z_test + sqrt_one_minus_alphas_cumprod_t * epsilon
        
        # Predict noise
        epsilon_pred = model(z_t, t_normalized)  # (B, C, D, H, W)
        
        # MSE error
        error = nn.functional.mse_loss(epsilon_pred, epsilon, reduction='none')  # (B, C, D, H, W)
        error = error.mean(dim=(1, 2, 3, 4))  # (B,)
        errors.append(error)
    
    # Stack: (n_samples, B)
    errors = torch.stack(errors, dim=0)
    
    # Aggregate: mean over n_samples
    scores = errors.mean(dim=0)  # (B,)
    
    return scores


@torch.no_grad()
def compute_anomaly_score_ddpm_small_t(
    model: nn.Module,
    z_test: torch.Tensor,
    noise_schedule: NoiseSchedule,
    t_max: int = 100,
    n_samples: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    DDPM Anomaly Score (Small t only).
    
    작은 t 구간에서만 오차를 계산 (원래 수식: S_Diff(z) = Σ_{t ∈ T_small} ||ε - ε_θ(z_t, t)||²)
    
    Args:
        model: Trained DDPM model
        z_test: (B, C, D, H, W) or (C, D, H, W) test latent
        noise_schedule: NoiseSchedule object
        t_max: Maximum timestep to sample from (small t only)
        n_samples: Number of timesteps to sample
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
    
    z_test = z_test.to(device)
    
    # Sample from small t range only
    errors = []
    for _ in range(n_samples):
        # Sample random timestep from [0, t_max)
        t = torch.randint(0, min(t_max, noise_schedule.timesteps), (batch_size,), device=device).long()
        t_normalized = t.float() / noise_schedule.timesteps
        
        # Sample noise
        epsilon = torch.randn_like(z_test)
        
        # Get noise schedule values
        sqrt_alphas_cumprod_t = noise_schedule.sqrt_alphas_cumprod[t]  # (B,)
        sqrt_one_minus_alphas_cumprod_t = noise_schedule.sqrt_one_minus_alphas_cumprod[t]  # (B,)
        
        # Expand for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None, None]
        
        # Forward diffusion
        z_t = sqrt_alphas_cumprod_t * z_test + sqrt_one_minus_alphas_cumprod_t * epsilon
        
        # Predict noise
        epsilon_pred = model(z_t, t_normalized)  # (B, C, D, H, W)
        
        # MSE error
        error = nn.functional.mse_loss(epsilon_pred, epsilon, reduction='none')  # (B, C, D, H, W)
        error = error.mean(dim=(1, 2, 3, 4))  # (B,)
        errors.append(error)
    
    # Stack: (n_samples, B)
    errors = torch.stack(errors, dim=0)
    
    # Aggregate: mean over n_samples (equivalent to sum then average)
    scores = errors.mean(dim=0)  # (B,)
    
    return scores


if __name__ == "__main__":
    # 간단한 테스트
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from latent_velocity_estimator import LatentVelocityEstimator
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Noise schedule (on device)
    noise_schedule = NoiseSchedule(timesteps=1000, schedule_type="cosine", device=device)
    print(f"Noise schedule timesteps: {noise_schedule.timesteps}")
    print(f"Beta range: [{noise_schedule.betas.min():.4f}, {noise_schedule.betas.max():.4f}]")
    
    # 모델 생성 (Flow Matching과 동일한 구조 재사용)
    model = LatentVelocityEstimator(
        in_channels=128,
        base_channels=64,
    ).to(device)
    
    # Test data
    batch_size = 2
    z_0 = torch.randn(batch_size, 128, 8, 16, 16).to(device)
    
    # Test DDPM loss
    print("\n--- Testing DDPM Loss ---")
    loss, info = compute_ddpm_loss(model, z_0, noise_schedule, device)
    print(f"Loss: {loss.item():.4f}")
    print(f"z_t shape: {info['z_t'].shape}")
    print(f"epsilon shape: {info['epsilon'].shape}")
    print(f"epsilon_pred shape: {info['epsilon_pred'].shape}")
    
    # Test anomaly score
    print("\n--- Testing Anomaly Score (DDPM) ---")
    z_test = torch.randn(batch_size, 128, 8, 16, 16).to(device)
    scores = compute_anomaly_score_ddpm(model, z_test, noise_schedule, n_samples=5, device=device)
    print(f"Anomaly scores: {scores}")
    
    # Test anomaly score (small t)
    print("\n--- Testing Anomaly Score (Small t) ---")
    scores_small_t = compute_anomaly_score_ddpm_small_t(
        model, z_test, noise_schedule, t_max=100, n_samples=5, device=device
    )
    print(f"Anomaly scores (small t): {scores_small_t}")
    
    print("\n✅ All tests passed!")
