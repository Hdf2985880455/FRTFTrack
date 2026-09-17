import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class CompressionEncoder(nn.Module):
    """
    CI2P compression encoder supporting both 2x and 4x spatial compression.
    """

    def __init__(self, in_chans=3, compress_dim=3, compression_ratio=2):
        super().__init__()
        self.compression_ratio = compression_ratio

        if compression_ratio == 2:
            self.compress_encoder = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    compress_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.GroupNorm(1, compress_dim),
                nn.ReLU(inplace=True),
            )
        elif compression_ratio == 4:
            self.compress_encoder = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    compress_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.GroupNorm(1, compress_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    compress_dim,
                    compress_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=True,
                ),
                nn.GroupNorm(1, compress_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unsupported compression_ratio={compression_ratio}")

        self._init_weights()
        self.init_rgb_lowpass()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def init_rgb_lowpass(self):
        if self.compression_ratio != 2:
            return

        conv = self.compress_encoder[0]
        if not isinstance(conv, nn.Conv2d):
            return
        if conv.in_channels != 3 or conv.out_channels != 3:
            return

        with torch.no_grad():
            conv.weight.zero_()
            kernel = conv.weight.new_tensor([
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ]) / 16.0
            for c in range(3):
                conv.weight[c, c].copy_(kernel)
            if conv.bias is not None:
                conv.bias.zero_()

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, compress_dim, H//compression_ratio, W//compression_ratio)
        """
        return self.compress_encoder(x)


class CI2PEmbed(nn.Module):
    """
    CI2P patch embedding supporting both 2x and 4x compression.

    For patch_size=16:
    - baseline PatchEmbed: H/16 x W/16
    - 2x CI2P: H/2 -> kernel/stride 8 => H/16 x W/16
    - 4x CI2P: H/4 -> kernel/stride 8 => H/32 x W/32
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        use_compression=True,
        compression_dim=64,
        compression_ratio=4,
        init_from_patch_embed=False,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten
        self.use_compression = use_compression
        self.downsample_ratio = compression_ratio
        self.init_from_patch_embed = init_from_patch_embed

        if use_compression:
            if patch_size[0] % 2 != 0 or patch_size[1] % 2 != 0:
                raise ValueError(
                    f"CI2PEmbed requires even patch_size when use_compression=True, got {patch_size}"
                )

            self.compression_encoder = CompressionEncoder(
                in_chans=in_chans,
                compress_dim=compression_dim,
                compression_ratio=compression_ratio,
            )

            # 压缩后再用 patch_size//2 投影
            self.effective_patch_size = (
                patch_size[0] // 2,
                patch_size[1] // 2,
            )

            self.proj = nn.Conv2d(
                compression_dim,
                embed_dim,
                kernel_size=self.effective_patch_size,
                stride=self.effective_patch_size,
                bias=True,
            )

            # 这里只作为初始化参考；真实运行时可通过 get_output_hw 适配变分辨率输入
            self.grid_size = self.get_output_hw(img_size[0], img_size[1])

        else:
            self.compression_encoder = None
            self.effective_patch_size = patch_size

            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )

            self.grid_size = (
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            )

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self._init_proj()

    def get_patch_stride(self):
        """
        Effective stride (in input pixels) per token along one axis.

        Standard PatchEmbed: stride == patch_size
        2x CI2P: stride == 2 * (patch_size // 2) == patch_size
        4x CI2P: stride == 4 * (patch_size // 2) == 2 * patch_size
        """
        if self.use_compression and self.compression_encoder is not None:
            return self.downsample_ratio * self.effective_patch_size[0]
        return self.patch_size[0]

    def _init_proj(self):
        if isinstance(self.proj, nn.Conv2d):
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
            if self.proj.bias is not None:
                nn.init.constant_(self.proj.bias, 0.0)

    def get_output_hw(self, H, W):
        """
        返回真实输入分辨率对应的 token grid 大小。

        CI2P:
            H_tok = (H // downsample_ratio) // (patch_size // 2)
        """
        if self.use_compression and self.compression_encoder is not None:
            out_h = (H // self.downsample_ratio) // self.effective_patch_size[0]
            out_w = (W // self.downsample_ratio) // self.effective_patch_size[1]
        else:
            out_h = H // self.patch_size[0]
            out_w = W // self.patch_size[1]
        return out_h, out_w

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            flatten=True  -> (B, N, C)
            flatten=False -> (B, C, H', W')
        """
        if self.use_compression and self.compression_encoder is not None:
            x = self.compression_encoder(x)
            x = self.proj(x)
        else:
            x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        x = self.norm(x)
        return x
