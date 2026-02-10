import torch
from pathlib import Path
import numpy as np

# pip install netcdf4 h5netcdf s3fs
if __name__ == "__main__":
    for ds_dir, split in zip(
            (Path("/scratch/dwcgt/original_dataset/"), Path("/scratch/dwcgt/original_dataset/test/")),
            ("train", "test")
    ):
        # shards = []
        # for p in ds_dir.glob("*pt"):
        #     traj = torch.load(p)
        #     T, C, H, W = traj.shape
        #     shards.append(traj)
        # ds = torch.stack(shards, dim=0)
        ds = torch.load(ds_dir / f"{split}.pt").to(torch.float32)
        N, T, C, H, W = ds.shape
        residuals = ds[:, 16:] - ds[:, :-16]
        std_residual = residuals.std()
        time = torch.tensor([t_idx * 0.007012483601762931 for t_idx in range(T)], dtype=torch.float64)
        ds_write = {
            "velocity_std_residual": std_residual,
            "time": time,
            "velocity": ds,
        }
        torch.save(ds_write, ds_dir / f"{split}_ds.pt")

