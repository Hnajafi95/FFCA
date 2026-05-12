"""SRDRN — PyTorch port + HDF5 weight loader.

Mirrors the TF/Keras `Network.py:Generator` architecture so we can run FFCA
on the existing `Original_SRDRN_epoch_160` checkpoint without re-training.

Architecture (input: B × 6 × H × W):
  conv0 (3×3, 6→64), PReLU(64)
  16× ResBlock { conv(3×3, 64→64), BN, PReLU(64), conv(3×3, 64→64), BN, + skip }
  conv_post (3×3, 64→64), BN, + initial-skip
  conv_up1 (3×3, 64→512), Upsample×2, PReLU(512)
  conv_up2 (3×3, 512→512), Upsample×3, PReLU(512)
  conv_up3 (3×3, 512→512), Upsample×2, PReLU(512)
  Dropout2d(0.1)
  conv_out (9×9, 512→1)

Output upscaling: H, W → H·12, W·12.

NOTE: Keras' Conv2D kernel layout is (kH, kW, Cin, Cout); PyTorch is
(Cout, Cin, kH, kW). We transpose on load. Keras BN order is
(gamma, beta, mean, var) which matches PyTorch (weight, bias, running_mean,
running_var). Keras PReLU with shared_axes=[1,2] keeps alpha shape
(1, 1, C); PyTorch nn.PReLU(num_parameters=C) is the equivalent.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch, momentum=0.5)
        self.act = nn.PReLU(num_parameters=ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch, momentum=0.5)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return x + y


class _UpBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, factor: int):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.factor = factor
        self.act = nn.PReLU(num_parameters=ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        return self.act(x)


class SRDRN(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        # Stem
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.act0 = nn.PReLU(num_parameters=64)
        # 16 residual blocks
        self.blocks = nn.ModuleList([_ResBlock(64) for _ in range(16)])
        # Post-residual
        self.conv_post = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_post = nn.BatchNorm2d(64, momentum=0.5)
        # Upsampling stack: x2, x3, x2  =>  total ×12
        self.up1 = _UpBlock(64, 512, factor=2)
        self.up2 = _UpBlock(512, 512, factor=3)
        self.up3 = _UpBlock(512, 512, factor=2)
        self.dropout = nn.Dropout2d(p=0.1)
        self.conv_out = nn.Conv2d(512, 1, 9, padding=4)

    def forward(self, x):
        x0 = self.act0(self.conv0(x))
        h = x0
        for blk in self.blocks:
            h = blk(h)
        h = self.bn_post(self.conv_post(h))
        h = h + x0  # initial skip
        h = self.up1(h); h = self.up2(h); h = self.up3(h)
        h = self.dropout(h)
        return self.conv_out(h)


def _tf_to_torch_kernel(tf_kernel: np.ndarray) -> np.ndarray:
    """Keras (kH, kW, Cin, Cout) → PyTorch (Cout, Cin, kH, kW)."""
    return np.transpose(tf_kernel, (3, 2, 0, 1))


def load_srdrn_from_h5(h5_path: str | Path, in_channels: int = 6) -> SRDRN:
    """Build a PyTorch SRDRN and load weights from a Keras HDF5 checkpoint."""
    model = SRDRN(in_channels=in_channels)
    f = h5py.File(str(h5_path), "r")
    W = f["model_weights"]

    def conv(name):
        g = W[name][name]
        return _tf_to_torch_kernel(g["kernel:0"][()]), g["bias:0"][()]

    def bn(name):
        g = W[name][name]
        return (g["gamma:0"][()], g["beta:0"][()],
                g["moving_mean:0"][()], g["moving_variance:0"][()])

    def prelu(name):
        g = W[name][name]
        alpha = g["alpha:0"][()]   # shape (1, 1, C)
        return alpha.reshape(-1)

    with torch.no_grad():
        # Stem
        k, b = conv("conv2d")
        model.conv0.weight.copy_(torch.from_numpy(k))
        model.conv0.bias.copy_(torch.from_numpy(b))
        model.act0.weight.copy_(torch.from_numpy(prelu("p_re_lu")))

        # 16 residual blocks
        # Keras numbering: block i uses conv2d_{2i+1}, conv2d_{2i+2},
        #                  batch_normalization_{2i}, batch_normalization_{2i+1},
        #                  p_re_lu_{i+1}.
        for i in range(16):
            k1, b1 = conv(f"conv2d_{2*i+1}")
            k2, b2 = conv(f"conv2d_{2*i+2}")
            blk = model.blocks[i]
            blk.conv1.weight.copy_(torch.from_numpy(k1))
            blk.conv1.bias.copy_(torch.from_numpy(b1))
            blk.conv2.weight.copy_(torch.from_numpy(k2))
            blk.conv2.bias.copy_(torch.from_numpy(b2))

            bn_a_name = "batch_normalization" if i == 0 else f"batch_normalization_{2*i}"
            # Block i uses BN_{2i} and BN_{2i+1}. BN_0 in HDF5 is "batch_normalization"
            # (no suffix); BN_1 is "batch_normalization_1"; etc.
            def _bn_key(idx):
                return "batch_normalization" if idx == 0 else f"batch_normalization_{idx}"
            g1, bt1, m1, v1 = bn(_bn_key(2*i))
            g2, bt2, m2, v2 = bn(_bn_key(2*i + 1))
            blk.bn1.weight.copy_(torch.from_numpy(g1)); blk.bn1.bias.copy_(torch.from_numpy(bt1))
            blk.bn1.running_mean.copy_(torch.from_numpy(m1)); blk.bn1.running_var.copy_(torch.from_numpy(v1))
            blk.bn2.weight.copy_(torch.from_numpy(g2)); blk.bn2.bias.copy_(torch.from_numpy(bt2))
            blk.bn2.running_mean.copy_(torch.from_numpy(m2)); blk.bn2.running_var.copy_(torch.from_numpy(v2))

            blk.act.weight.copy_(torch.from_numpy(prelu(f"p_re_lu_{i+1}")))

        # Post-residual: conv2d_33 + BN_32
        k, b = conv("conv2d_33")
        model.conv_post.weight.copy_(torch.from_numpy(k))
        model.conv_post.bias.copy_(torch.from_numpy(b))
        g, bt, m, v = bn("batch_normalization_32")
        model.bn_post.weight.copy_(torch.from_numpy(g)); model.bn_post.bias.copy_(torch.from_numpy(bt))
        model.bn_post.running_mean.copy_(torch.from_numpy(m)); model.bn_post.running_var.copy_(torch.from_numpy(v))

        # Upsample 1: conv2d_34 + p_re_lu_17
        k, b = conv("conv2d_34")
        model.up1.conv.weight.copy_(torch.from_numpy(k))
        model.up1.conv.bias.copy_(torch.from_numpy(b))
        model.up1.act.weight.copy_(torch.from_numpy(prelu("p_re_lu_17")))

        # Upsample 2: conv2d_35 + p_re_lu_18
        k, b = conv("conv2d_35")
        model.up2.conv.weight.copy_(torch.from_numpy(k))
        model.up2.conv.bias.copy_(torch.from_numpy(b))
        model.up2.act.weight.copy_(torch.from_numpy(prelu("p_re_lu_18")))

        # Upsample 3: conv2d_36 + p_re_lu_19
        k, b = conv("conv2d_36")
        model.up3.conv.weight.copy_(torch.from_numpy(k))
        model.up3.conv.bias.copy_(torch.from_numpy(b))
        model.up3.act.weight.copy_(torch.from_numpy(prelu("p_re_lu_19")))

        # Final 9×9 conv
        k, b = conv("conv2d_37")
        model.conv_out.weight.copy_(torch.from_numpy(k))
        model.conv_out.bias.copy_(torch.from_numpy(b))

    f.close()
    model.eval()
    return model


if __name__ == "__main__":
    import sys
    h5 = ("/Users/hnaja002/Documents/projects/FFCA/FFCA_dump/"
          "FFCA_archetype_dynamic/claude_playground/SRDRN_Files/"
          "Original_SRDRN_epoch_160")
    m = load_srdrn_from_h5(h5)
    print(f"SRDRN loaded: {sum(p.numel() for p in m.parameters()):,} parameters")
    # forward sanity: input 11x13x6 from Phase 2.1
    x = torch.randn(1, 6, 11, 13)
    y = m(x)
    print(f"forward {tuple(x.shape)} → {tuple(y.shape)}  (expect 1×1×132×156)")
