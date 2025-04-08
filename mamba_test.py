import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

from Mamba_ import Mamba


class MambaLayer(nn.Module):
    def __init__(self,in_dim, dim, d_state=16, d_conv=4, expand=2, num_slices_small=None):
        super().__init__()


        self.down =nn.Sequential(
        nn.Conv3d(in_dim, dim, kernel_size=7, stride=2, padding=3),
    )
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2",
            nslices_small=num_slices_small,
        )

    def forward(self, x):
        x = self.down(x)
        B, C = x.shape[:2]
        x_skip = x.cuda()
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat = x_flat.cuda()
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        return out



if __name__ == "__main__":
    img = torch.ones([1, 4, 128, 128, 128]).cuda()
    dims = [4,4, 32, 64, 128]
    model = MambaLayer(in_dim=dims[0],dim=dims[1])
    model = model.cuda()

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
