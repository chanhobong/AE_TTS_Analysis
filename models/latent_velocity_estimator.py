"""
latent_velocity_estimator.py
- LatentVelocityEstimator: 3D Convolution 기반 ResNet 모델
- Time Conditioning을 위한 Time Embedding
- Flow Matching을 위한 Velocity 예측 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """
    Time step t를 embedding으로 변환.
    Sinusoidal positional encoding 사용.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar time steps in [0, 1]
        Returns:
            emb: (B, dim) time embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        return emb


class TimeMLP(nn.Module):
    """
    Time embedding을 채널 수로 확장하는 MLP.
    """
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_emb: (B, time_dim) time embeddings
        Returns:
            out: (B, out_dim) expanded embeddings
        """
        return self.mlp(t_emb)


class ResidualBlock3D(nn.Module):
    """
    3D Convolution 기반 Residual Block with Time Conditioning.
    """
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, 
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        
        # Time conditioning MLP
        self.time_mlp = TimeMLP(time_dim, out_channels)
        
        # Conv layers
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(8, out_channels)  # GroupNorm for stability
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 
                               padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.skip = nn.Identity()
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) input features
            t_emb: (B, time_dim) time embeddings
        Returns:
            out: (B, C_out, D', H', W') output features
        """
        # Time conditioning
        t_emb_expanded = self.time_mlp(t_emb)  # (B, out_channels)
        # Add channel dimension for broadcasting: (B, C, 1, 1, 1)
        t_emb_expanded = t_emb_expanded[:, :, None, None, None]
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)  # SiLU activation
        h = h + t_emb_expanded  # Time conditioning
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Skip connection
        skip = self.skip(x)
        out = F.silu(h + skip)
        
        return out


class LatentVelocityEstimator(nn.Module):
    """
    3D Convolution 기반 ResNet 모델 for Flow Matching.
    
    입력: z_t (B, C, D, H, W)와 t (B,) scalar
    출력: v_θ(z_t, t) - velocity field (B, C, D, H, W)
    """
    def __init__(
        self,
        in_channels: int = 128,
        base_channels: int = 64,
        time_dim: int = 128,
        num_res_blocks: int = 4,
    ):
        """
        Args:
            in_channels: 입력 latent의 채널 수 (예: 128)
            base_channels: 기본 채널 수 (64~128 추천, M5 맥북 고려)
            time_dim: Time embedding 차원
            num_res_blocks: ResBlock 개수
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        # Input projection
        self.input_conv = nn.Conv3d(in_channels, base_channels, 
                                    kernel_size=3, padding=1)
        self.input_norm = nn.GroupNorm(8, base_channels)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock3D(channels, channels, time_dim)
            )
        
        # Output projection (back to input channels)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv3d(base_channels, in_channels, 
                                     kernel_size=3, padding=1)
        
    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, C, D, H, W) latent at time t
            t: (B,) time steps in [0, 1]
        Returns:
            v_t: (B, C, D, H, W) predicted velocity field
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_dim)
        
        # Input projection
        h = self.input_conv(z_t)
        h = self.input_norm(h)
        h = F.silu(h)
        
        # Residual blocks
        for res_block in self.res_blocks:
            h = res_block(h, t_emb)
        
        # Output projection
        h = self.output_norm(h)
        h = F.silu(h)
        v_t = self.output_conv(h)
        
        return v_t


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 개수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 간단한 테스트
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 생성
    model = LatentVelocityEstimator(
        in_channels=128,
        base_channels=64,
        time_dim=128,
        num_res_blocks=4,
    ).to(device)
    
    # 파라미터 개수
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Forward test
    batch_size = 2
    z_t = torch.randn(batch_size, 128, 8, 16, 16).to(device)
    t = torch.rand(batch_size).to(device)
    
    with torch.no_grad():
        v_t = model(z_t, t)
        print(f"Input shape: {z_t.shape}")
        print(f"Output shape: {v_t.shape}")
        print(f"✅ Forward pass successful!")
