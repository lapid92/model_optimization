import collections.abc
import math
from functools import partial
from itertools import repeat
from typing import Tuple, Union, Optional, Callable, List

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn, Tensor
from torch.nn import functional as F


@torch.fx.wrap
def zero_expand(out, x):
    out[:, 0, :] = x

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

class MyGELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        return input * self.sigmoid(1.702*input)


def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


class Mlp(nn.Module):
    """
        MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            # act_layer=nn.GELU,
            act_layer=MyGELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, 1, N, C)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.reshape(B, N, C)
        return x

class ModifiedAttention(nn.Module):
    """
    The ModifiedAttention class is derived from the timm/Attention class.
    We've adjusted the class to prevent folding on the batch axis and to refrain from performing matmul on tensors
    with more than 3 dimensions (considering the batch axis).
    Despite these modifications, the module retains its original functionality.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # [B, N, 3*C] --> [B, 1, N, 3*C]
        x = x.reshape(B, 1, N, C)
        # [B, 1, N, 3*C] --> [B, 1, N, 3*C]
        qkv = self.qkv(x)
        # [B, 1, N, 3*C] --> [B, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # [B, N, 3, num_heads, head_dim] --> [B, 3, num_heads, N, head_dim]
        qkv = qkv.permute(0, 2, 3, 1, 4)
        # [B, 3, num_heads, N, head_dim] --> 3 * [B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(1)

        # We've adjusted this section to calculate the attention individually for each head and patch.
        head_list = []

        # [B, num_heads, N, head_dim] --> num_heads * [B, N, head_dim]
        q_split = q.unbind(1)
        k_split = k.unbind(1)
        v_split = v.unbind(1)

        for head in range(self.num_heads):
            k_head = k_split[head]
            q_head = q_split[head]
            v_head = v_split[head]

            k_head = self.k_norm(k_head)
            q_head = self.q_norm(q_head)

            q_head = q_head * self.scale

            # [B, N, head_dim] --> [B, head_dim, N]
            k_head = k_head.transpose(-2, -1)

            # [B, N, head_dim] @ [B, head_dim, N] --> [B, N, N]
            attn_head = q_head @ k_head

            attn_head = attn_head.softmax(dim=-1)
            attn_head = self.attn_drop(attn_head)

            # [B, N, N] @ [B, N, head_dim] --> [B, N, head_dim]
            x_head = attn_head @ v_head

            # num_heads * [B, N, head_dim]
            head_list.append(x_head)

        # num_heads * [B, N, head_dim] --> [B, num_heads, N, head_dim]
        concat_heads = torch.stack(head_list, dim=1)

        # [B, num_heads, N, head_dim] --> [B, N, num_heads, head_dim]
        x = concat_heads.transpose(1, 2)

        # [B, N, num_heads, head_dim] --> [B, 1, N, C]
        x = x.reshape(B, 1, N, C)
        x = self.proj(x)
        # [B, 1, N, C] --> [B, N, C]
        x = x.reshape(B, N, C)
        x = self.proj_drop(x)
        return x


# class ModifiedAttention(nn.Module):
#     """
#     The ModifiedAttention class is derived from the timm/Attention class.
#     We've adjusted the class to prevent folding on the batch axis and to refrain from performing matmul on tensors
#     with more than 3 dimensions (considering the batch axis).
#     Despite these modifications, the module retains its original functionality.
#     """
#
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             attn_drop: float = 0.,
#             proj_drop: float = 0.,
#             norm_layer: nn.Module = nn.LayerNorm,
#     ) -> None:
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, N, C = x.shape
#         # [B, N, 3*C] --> [B, N, 3*C]
#         qkv = self.qkv(x)
#         # [B, N, 3*C] --> [B, N, 3, num_heads, head_dim]
#         qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
#         # [B, N, 3, num_heads, head_dim] --> [B, 3, num_heads, N, head_dim]
#         qkv = qkv.permute(0, 2, 3, 1, 4)
#         # [B, 3, num_heads, N, head_dim] --> 3 * [B, num_heads, N, head_dim]
#         q, k, v = qkv.unbind(1)
#
#         # We've adjusted this section to calculate the attention individually for each head and patch.
#         head_list = []
#
#         # [B, num_heads, N, head_dim] --> num_heads * [B, N, head_dim]
#         q_split = q.unbind(1)
#         k_split = k.unbind(1)
#         v_split = v.unbind(1)
#
#         for head in range(self.num_heads):
#             k_head = k_split[head]
#             q_head = q_split[head]
#             v_head = v_split[head]
#
#             k_head = self.k_norm(k_head)
#             q_head = self.q_norm(q_head)
#
#             q_head = q_head * self.scale
#
#             # [B, N, head_dim] --> [B, head_dim, N]
#             k_head = k_head.transpose(-2, -1)
#
#             # [B, N, head_dim] @ [B, head_dim, N] --> [B, N, N]
#             attn_head = q_head @ k_head
#
#             attn_head = attn_head.softmax(dim=-1)
#             attn_head = self.attn_drop(attn_head)
#
#             # [B, N, N] @ [B, N, head_dim] --> [B, N, head_dim]
#             x_head = attn_head @ v_head
#
#             # num_heads * [B, N, head_dim]
#             head_list.append(x_head)
#
#         # num_heads * [B, N, head_dim] --> [B, num_heads, N, head_dim]
#         concat_heads = torch.stack(head_list, dim=1)
#
#         # [B, num_heads, N, head_dim] --> [B, N, num_heads, head_dim]
#         x = concat_heads.transpose(1, 2)
#
#         # [B, N, num_heads, head_dim] --> [B, N, C]
#         x = x.reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            # act_layer: nn.Module = nn.GELU,
            act_layer: nn.Module = MyGELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ModifiedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size: Optional[Union[int, Tuple[int, int]]] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
            strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # TODO:
        # check if need to change
        x = self.proj(x)
        B, C, H, W = x.shape
        # BCHW -> BCL, L = H * W
        x = x.reshape(B, C, H*W)
        # x = x.flatten(2)
        # BCL -> BLC
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class VitTiny(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        img_size = (224, 224)
        patch_size = 16
        in_chans = 3
        embed_dim = 192
        self.global_pool = 'token'

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
        )

        num_patches = 196
        reduction = 16
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = None
        self.num_prefix_tokens = 1
        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        depth = 12
        num_heads = 3
        num_classes = 1000
        self.num_classes = num_classes
        mlp_ratio = 4.0
        proj_drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_rate = 0.0
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # act_layer = nn.GELU
        act_layer = MyGELU
        mlp_layer = Mlp

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim)

        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # We have defined pretrained_cfg to represent the configuration specific to vit pretrained model,
        # including relevant items for dataset and data loader.
        self.pretrained_cfg = {
            'architecture': 'vit_tiny_patch16_224',
            'tag': 'augreg_in21k_ft_in1k',
            'custom_load': True,
            'input_size': (3, 224, 224),
            'fixed_input_size': True,
            'interpolation': 'bicubic',
            'crop_pct': 0.9,
            'crop_mode': 'center',
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
            'num_classes': 1000,
            'pool_size': None,
            'first_conv': 'patch_embed.proj',
            'classifier': 'head'}

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed

        to_cat = [self.cls_token.expand(x.shape[0], -1, -1)]

        # B, C, L = x.shape
        # to_cat = [self.cls_token.expand(B, 1, L)]

        # to_cat_tensor = []

        # for i in range(B):
        #     to_cat_tensor.append(self.cls_token.squeeze(0))
        # to_cat = [torch.stack(to_cat_tensor, dim=0)]
        # to_cat = [self.cls_token.repeat((B, 1, 1))]
        # zero_tensor = x.clone().zero_()[:, 0:1, :]
        # zero_expand(zero_tensor, self.cls_token)
        # zero_tensor[:, 0, :] = self.cls_token
        # to_cat = [zero_tensor]
        x = torch.cat(to_cat + [x], dim=1)
<<<<<<< HEAD
<<<<<<< HEAD
        # x = torch.cat([self.cls_token] + [x], dim=1)
=======
>>>>>>> Vit tiny first commit
=======
        # x = torch.cat([self.cls_token] + [x], dim=1)
>>>>>>> First Vit tiny push
        x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def save_pretrained(self, save_directory, **kwargs):
        # Call the original save_pretrained method
        super().save_pretrained(save_directory, **kwargs)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> First Vit tiny push

class VitTiny_no_cls_token(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        img_size = (224, 224)
        patch_size = 16
        in_chans = 3
        embed_dim = 192
        self.global_pool = 'avg'

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True,
        )

        num_patches = 196
        reduction = 16
        self.cls_token = None
        self.reg_token = None
        self.num_prefix_tokens = 0
        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=0.0)
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        depth = 12
        num_heads = 3
        num_classes = 1000
        self.num_classes = num_classes
        mlp_ratio = 4.0
        proj_drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_rate = 0.0
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # act_layer = nn.GELU
        act_layer = MyGELU
        mlp_layer = Mlp

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim)

        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # We have defined pretrained_cfg to represent the configuration specific to vit pretrained model,
        # including relevant items for dataset and data loader.
        self.pretrained_cfg = {
            'architecture': 'vit_tiny_patch16_224',
            'tag': 'augreg_in21k_ft_in1k',
            'custom_load': True,
            'input_size': (3, 224, 224),
            'fixed_input_size': True,
            'interpolation': 'bicubic',
            'crop_pct': 0.9,
            'crop_mode': 'center',
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
            'num_classes': 1000,
            'pool_size': None,
            'first_conv': 'patch_embed.proj',
            'classifier': 'head'}

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x[:, self.num_prefix_tokens:].mean(dim=1)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def save_pretrained(self, save_directory, **kwargs):
        # Call the original save_pretrained method
        super().save_pretrained(save_directory, **kwargs)
<<<<<<< HEAD
=======
>>>>>>> Vit tiny first commit
=======
>>>>>>> First Vit tiny push
