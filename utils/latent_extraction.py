"""
latent_extraction.py
- Extract spatial latent vectors from trained AE encoder
- Common utilities for DDPM, Flow Matching, and other latent-based models
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    """
    Dataset that provides spatial latent vectors from AE encoder.
    
    This extracts latents BEFORE pooling (spatial structure preserved).
    Used for DDPM, Flow Matching, etc.
    """
    def __init__(self, ae_model: nn.Module, volume_dataset: Dataset, device: torch.device):
        """
        Args:
            ae_model: Trained Autoencoder model (must have .enc attribute)
            volume_dataset: CTROIVolumeDataset
            device: Device for inference
        """
        self.ae_model = ae_model
        self.volume_dataset = volume_dataset
        self.device = device
        
        # Extract all latents (spatial, not pooled)
        print(f"Extracting spatial latents from {len(volume_dataset)} samples...")
        self.latents = []
        
        self.ae_model.eval()
        with torch.no_grad():
            for idx in range(len(volume_dataset)):
                sample = volume_dataset[idx]
                x = sample["x"].unsqueeze(0).to(device)  # (1, 1, Z, Y, X)
                
                # Extract spatial latent (before pooling)
                z = self.ae_model.enc(x)  # (1, C, D, H, W)
                z = z.squeeze(0).cpu()  # (C, D, H, W)
                
                self.latents.append(z)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(volume_dataset)} samples...")
        
        print(f"✅ Extracted {len(self.latents)} spatial latents")
        if len(self.latents) > 0:
            print(f"   Latent shape: {self.latents[0].shape}")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        z = self.latents[idx]  # (C, D, H, W)
        return {"z": z}


def extract_spatial_latents_batch(
    ae_model: nn.Module,
    volume_dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
) -> list[torch.Tensor]:
    """
    Extract spatial latents in batches (more memory efficient for large datasets).
    
    Args:
        ae_model: Trained Autoencoder model
        volume_dataset: CTROIVolumeDataset
        device: Device for inference
        batch_size: Batch size for extraction
    
    Returns:
        List of latent tensors, each of shape (C, D, H, W)
    """
    from torch.utils.data import DataLoader
    
    ae_model.eval()
    latents = []
    
    loader = DataLoader(volume_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Extracting spatial latents from {len(volume_dataset)} samples (batch_size={batch_size})...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device)  # (B, 1, Z, Y, X)
            
            # Extract spatial latents
            z_batch = ae_model.enc(x)  # (B, C, D, H, W)
            
            # Move to CPU and split into individual samples
            z_batch = z_batch.cpu()
            for i in range(z_batch.shape[0]):
                latents.append(z_batch[i])  # (C, D, H, W)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(volume_dataset)} samples...")
    
    print(f"✅ Extracted {len(latents)} spatial latents")
    if len(latents) > 0:
        print(f"   Latent shape: {latents[0].shape}")
    
    return latents
