"""
测试CI2P模块的基本功能
"""

import os
import sys

# 把项目根目录加入 PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
from thop import profile

# ✅ 只从具体文件导入，避免触发 layers/__init__.py 的复杂导入链
from lib.models.layers.ci2p_embed import CI2PEmbed
from lib.models.layers.patch_embed import PatchEmbed


def test_ci2p_output_shape():
    """测试CI2P输出形状是否正确"""
    batch_size = 2
    img_size = 224
    patch_size = 16
    embed_dim = 768

    standard_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    ci2p_embed = CI2PEmbed(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
        use_compression=True, compression_dim=64
    )

    x = torch.randn(batch_size, 3, img_size, img_size)

    std_out = standard_embed(x)
    ci2p_out = ci2p_embed(x)

    print(f"Standard PatchEmbed output shape: {std_out.shape}")
    print(f"Standard num_patches: {standard_embed.num_patches}")
    print(f"CI2P Embed output shape: {ci2p_out.shape}")
    print(f"CI2P num_patches: {ci2p_embed.num_patches}")

    assert ci2p_embed.num_patches < standard_embed.num_patches, "CI2P should reduce number of patches"
    # 验证embedding维度一致
    assert std_out.shape[-1] == ci2p_out.shape[-1] == embed_dim, \
        f"Embedding dimension should be {embed_dim}, got std={std_out.shape[-1]}, ci2p={ci2p_out.shape[-1]}"

    # 验证batch size一致
    assert std_out.shape[0] == ci2p_out.shape[0] == batch_size, \
        f"Batch size should be {batch_size}, got std={std_out.shape[0]}, ci2p={ci2p_out.shape[0]}"

    # 验证压缩比
    expected_ratio = standard_embed.num_patches / ci2p_embed.num_patches
    assert abs(expected_ratio - 4.0) < 0.1, \
        f"Compression ratio should be ~4.0, got {expected_ratio}"
    print("✓ CI2P successfully reduces patch count")


def test_ci2p_flops():
    """使用 thop 统计 CI2P 与标准 PatchEmbed 的 MACs/FLOPs（推理口径）"""
    img_size = 224
    patch_size = 16
    embed_dim = 768

    standard_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    ci2p_embed = CI2PEmbed(
        img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
        use_compression=True, compression_dim=64
    )

    x = torch.randn(1, 3, img_size, img_size)

    standard_embed.eval()
    ci2p_embed.eval()

    with torch.no_grad():
        std_macs, std_params = profile(standard_embed, inputs=(x,), verbose=False)
        ci2p_macs, ci2p_params = profile(ci2p_embed, inputs=(x,), verbose=False)

    # 常见换算（论文里写清楚口径）：FLOPs ≈ 2 * MACs
    std_flops = 2 * std_macs
    ci2p_flops = 2 * ci2p_macs

    print(f"[PatchEmbed] MACs: {std_macs/1e9:.4f} G, FLOPs(≈2*MACs): {std_flops/1e9:.4f} G, Params: {std_params/1e6:.4f} M")
    print(f"[CI2PEmbed ] MACs: {ci2p_macs/1e9:.4f} G, FLOPs(≈2*MACs): {ci2p_flops/1e9:.4f} G, Params: {ci2p_params/1e6:.4f} M")
    print(f"Reduction (MACs): {(1 - ci2p_macs/std_macs) * 100:.2f}%")



def test_ci2p_forward_variations():
    """测试CI2P在不同输入尺寸和batch size下的前向传播"""
    embed_dim = 768
    patch_size = 16
    
    # 测试不同batch size
    for batch_size in [1, 2, 4, 8]:
        ci2p_embed = CI2PEmbed(
            img_size=224, patch_size=patch_size, embed_dim=embed_dim,
            use_compression=True, compression_dim=64
        )
        x = torch.randn(batch_size, 3, 224, 224)
        out = ci2p_embed(x)
        assert out.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {out.shape[0]}"
        assert out.shape[-1] == embed_dim, f"Embedding dim mismatch: expected {embed_dim}, got {out.shape[-1]}"
        print(f"✓ Batch size {batch_size}: output shape {out.shape}")
    
    # 测试不同输入尺寸
    for img_size in [224, 256, 384]:
        ci2p_embed = CI2PEmbed(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            use_compression=True, compression_dim=64
        )
        x = torch.randn(2, 3, img_size, img_size)
        out = ci2p_embed(x)
        expected_patches = ((img_size // 4) // (patch_size // 2)) ** 2
        assert out.shape[1] == expected_patches, \
            f"Patch count mismatch for img_size {img_size}: expected {expected_patches}, got {out.shape[1]}"
        print(f"✓ Image size {img_size}: {out.shape[1]} patches, output shape {out.shape}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing CI2P Module")
    print("=" * 60)
    
    print("\n[1] Testing output shape...")
    test_ci2p_output_shape()
    
    print("\n[2] Testing FLOPs...")
    test_ci2p_flops()
    
    print("\n[3] Testing forward propagation variations...")
    test_ci2p_forward_variations()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
