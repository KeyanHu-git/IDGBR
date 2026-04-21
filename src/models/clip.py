from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _to_2tuple(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-value tuple/list, got: {value!r}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer."""

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        act_cfg=dict(type="GELU"),
        ffn_drop=0.0,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = nn.GELU()  # Using nn.GELU directly

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # Padding to maintain the spatial size
            bias=True,
            groups=feedforward_channels,
        )

        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        self.drop = nn.Dropout(ffn_drop)

        self.layers = nn.Sequential(
            fc1, pe_conv, self.activate, self.drop, fc2, self.drop
        )

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, height, width)
        out = self.layers(x)

        out = x + out

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        #ret = super().forward(x.type(torch.float32))
        ret = super().forward(x)
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1q = LayerNorm(d_model)
        self.ln_1k = LayerNorm(d_model)
        self.ln_1v = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=q.dtype, device=q.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: list[torch.Tensor]):
        q, k, v = x[0], x[1], x[2]
        v = v + self.attention(self.ln_1q(q), self.ln_1k(k), self.ln_1v(v))
        v = v + self.mlp(self.ln_2(v))
        x = [q, v, v]
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        x = self.resblocks([q, k, v])
        # x = [ q k v ]
        value = x[2]
        return value


class QVisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        in_channels: int,
        cross_attn_dim: int = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        if cross_attn_dim is not None:
            self.conv1_k = nn.Conv2d(
                in_channels=cross_attn_dim,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            self.conv1_v = nn.Conv2d(
                in_channels=cross_attn_dim,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
        else:
            self.conv1_k = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            self.conv1_v = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )

        self.qtoken = nn.Parameter(
            torch.randn((input_resolution // patch_size) ** 2 + 1, 1, width)
        )

        self.ln_pre_q = LayerNorm(width)
        self.ln_pre_k = LayerNorm(width)
        self.ln_pre_v = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def x_pre(self, x: torch.Tensor, conv1: nn.Module, ln_pre: nn.Module):
        # x_shape=x.shape
        # if not x_shape[2] == self.input_resolution:
        #     x=F.interpolate(x,[self.input_resolution,self.input_resolution],mode='bilinear')
        x = conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        if q is None, the use qtoken replace q.
        """
        k = self.x_pre(k, ln_pre=self.ln_pre_k, conv1=self.conv1_k)
        v = self.x_pre(v, ln_pre=self.ln_pre_v, conv1=self.conv1_v)
        if q is None:
            qtoken = self.qtoken.expand(-1, k.shape[1], -1)
            q = qtoken
        else:
            q = self.x_pre(q, ln_pre=self.ln_pre_q, conv1=self.conv1_q)

        x = self.transformer(q, k, v)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 1:, :])
        # x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        in_channels: int,
        cross_attn_dim: int = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        if cross_attn_dim is not None:
            self.conv1_k = nn.Conv2d(
                in_channels=cross_attn_dim,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # self.conv1_v = nn.Conv2d(
            #     in_channels=cross_attn_dim,
            #     out_channels=width,
            #     kernel_size=patch_size,
            #     stride=patch_size,
            #     bias=False,
            # )
        else:
            self.conv1_k = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # self.conv1_v = nn.Conv2d(
            #     in_channels=in_channels,
            #     out_channels=width,
            #     kernel_size=patch_size,
            #     stride=patch_size,
            #     bias=False,
            # )

        scale = width**-0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre_q = LayerNorm(width)
        self.ln_pre_k = LayerNorm(width)
        # self.ln_pre_v = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def x_pre(self, x: torch.Tensor, conv1: nn.Module, ln_pre: nn.Module):
        # x_shape=x.shape
        # if not x_shape[2] == self.input_resolution:
        #     x=F.interpolate(x,[self.input_resolution,self.input_resolution],mode='bilinear')
        x = conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.x_pre(q, ln_pre=self.ln_pre_q, conv1=self.conv1_q)
        k = self.x_pre(k, ln_pre=self.ln_pre_k, conv1=self.conv1_k)
        # v = self.x_pre(v, ln_pre=self.ln_pre_v, conv1=self.conv1_v)
        v=k

        x = self.transformer(q, k, v)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 1:, :])
        # x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_block(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        in_channels: int,
        cross_attn_dim: int = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        if cross_attn_dim is not None:
            self.conv1_k = nn.Conv2d(
                in_channels=cross_attn_dim,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # self.conv1_v = nn.Conv2d(in_channels=cross_attn_dim, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        else:
            self.conv1_k = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # self.conv1_v = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        # self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2, width)
        )
        self.ln_pre_q = LayerNorm(width)
        self.ln_pre_k = LayerNorm(width)
        # self.ln_pre_v = LayerNorm(width)

        self.transformer_self_q = Transformer(width, layers, heads)
        # self.transformer_self_kv = Transformer(width, layers, heads)
        self.transformer_cross = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def x_pre(self, x: torch.Tensor, conv1: nn.Module, ln_pre: nn.Module):
        # x_shape=x.shape
        # if not x_shape[2] == self.input_resolution:
        #     x=F.interpolate(x,[self.input_resolution,self.input_resolution],mode='bilinear')
        x = conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        N, C, H, W = k.shape
        q = self.x_pre(q, ln_pre=self.ln_pre_q, conv1=self.conv1_q)
        k = self.x_pre(k, ln_pre=self.ln_pre_k, conv1=self.conv1_k)
        # v=self.x_pre(v,ln_pre=self.ln_pre_v,conv1=self.conv1_v)

        q += self.transformer_self_q(q, q, q)

        # k += self.transformer_self_kv(k,k,k)

        v = k
        # TODO: cross attention
        x = self.transformer_cross(q, k, v)

        x = x + q

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        # x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj

        x = x.permute(0, 2, 1).view(N, C, H, W)

        return x

class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        l,n,d = x.shape
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale.unsqueeze(0).repeat([l,1,1])) + shift.unsqueeze(0).repeat([l,1,1])
        return x
    
class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        cross_dim:int,
        temb_channels: int = 1280,
        patch_size: int = 1,
        width: int = 512,
        layers: int = 1,
        heads: int = 8,
        input_resolution=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        patch_h, patch_w = _to_2tuple(patch_size)
        self.patch_size = (patch_h, patch_w)

        self.conv1_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.conv1_k = nn.Conv2d(
            in_channels=cross_dim,
            out_channels=width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.ln_pre_q = LayerNorm(width)
        self.ln_pre_k = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.input_resolution = _to_2tuple(input_resolution) if input_resolution is not None else self.patch_size
        self.base_grid_size = (
            max(1, self.input_resolution[0] // self.patch_size[0]),
            max(1, self.input_resolution[1] // self.patch_size[1]),
        )
        
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.base_grid_size[0] * self.base_grid_size[1], width)
        )
        self.proj = nn.Conv2d(
            in_channels=width,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def _get_positional_embedding(self, grid_h: int, grid_w: int, dtype, device):
        pos = self.positional_embedding
        if (grid_h, grid_w) != self.base_grid_size:
            pos = pos.reshape(
                self.base_grid_size[0], self.base_grid_size[1], -1
            ).permute(2, 0, 1).unsqueeze(0)
            pos = F.interpolate(
                pos.float(),
                size=(grid_h, grid_w),
                mode="bicubic",
                align_corners=False,
            )
            pos = pos.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, -1)
        return pos.to(device=device, dtype=dtype)

    def x_pre(self, x: torch.Tensor, conv1: nn.Module, ln_pre: nn.Module):
        x = conv1(x)  # shape = [*, width, grid, grid]
        grid_h, grid_w = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x + self._get_positional_embedding(grid_h, grid_w, x.dtype, x.device)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = ln_pre(x)
        return x, grid_h, grid_w

    def forward(self, q: torch.Tensor,k: torch.Tensor):
        N = q.shape[0]
        res=q
        q, grid_h, grid_w = self.x_pre(q, ln_pre=self.ln_pre_q, conv1=self.conv1_q)
        k, _, _ = self.x_pre(k, ln_pre=self.ln_pre_k, conv1=self.conv1_k)
        
        x = self.transformer(q,k,k)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        x = x.permute(0, 2, 1).view(N, x.shape[2], grid_h, grid_w)

        out = self.proj(x)
        
        out = out + res

        return out
    
class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        patch_size: int = 1,
        width: int = 512,
        layers: int = 1,
        heads: int = 8,
        input_resolution=None,
    ):
        super().__init__()
        self.conv1_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.ln_pre_q = LayerNorm(width)

        self.transformer_self_q = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.input_resolution = input_resolution
        
        scale = width**-0.5
        self.positional_embedding = nn.Parameter(
                    scale * torch.randn((self.input_resolution // patch_size) ** 2, width)
                )
        self.proj = nn.Conv2d(
            in_channels=width,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def x_pre(self, x: torch.Tensor, conv1: nn.Module, ln_pre: nn.Module):
        x = conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def forward(self, q: torch.Tensor):
        N, C, H, W = q.shape
        x = self.x_pre(q, ln_pre=self.ln_pre_q, conv1=self.conv1_q)

        x = self.transformer_self_q(x,x,x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        x = x.permute(0, 2, 1).view(N, x.shape[2], H, W)

        out = self.proj(x)

        return out
    
class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        act_cfg=dict(type="GELU"),
        ffn_drop=0.0,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = nn.GELU()  # Using nn.GELU directly

        in_channels = embed_dims
        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        self.drop = nn.Dropout(ffn_drop)

        self.layers = nn.Sequential(
            fc1, self.activate, self.drop, fc2, self.drop
        )

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, height, width)
        out = self.layers(x)

        out = x + out

        return out


class MultiScaleConvAttn(nn.Module):
    def __init__(
        self,
        embed_dims
    ):
        super().__init__()

        in_channels = embed_dims
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv_m = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, height, width)
        x = self.gelu(x)
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x_m = x + x1 + x2 + x3
        x_m = self.conv_m(x_m)
        out = torch.mul(x_m, x)
        return out


class VisionTransformer_mixffn(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        in_channels: int,
        cross_attn_dim: int = None,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1_q = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        if cross_attn_dim is not None:
            self.conv1_k = nn.Conv2d(
                in_channels=cross_attn_dim,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # self.conv1_v = nn.Conv2d(in_channels=cross_attn_dim, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        else:
            self.conv1_k = nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
            # self.conv1_v = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.ln_pre_q = LayerNorm(width)
        self.ln_pre_k = LayerNorm(width)
        self.ln_pre_v = LayerNorm(width)

        self.transformer_self_q = Transformer(width, layers, heads)
        self.transformer_cross = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)

        self.mixffn = MixFFN(embed_dims=width, feedforward_channels=width * 4)
        self.proj = nn.Conv2d(
            in_channels=width,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def x_pre(self, x: torch.Tensor, conv1: nn.Module, ln_pre: nn.Module):
        # x_shape=x.shape
        # if not x_shape[2] == self.input_resolution:
        #     x=F.interpolate(x,[self.input_resolution,self.input_resolution],mode='bilinear')
        x = conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        x = ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor = None):
        N, C, H, W = q.shape
        q = self.x_pre(q, ln_pre=self.ln_pre_q, conv1=self.conv1_q)
        k = self.x_pre(k, ln_pre=self.ln_pre_k, conv1=self.conv1_k)

        q += self.transformer_self_q(q, q, q)

        v = k
        # TODO: cross attention
        x = self.transformer_cross(q, k, v)

        x = x + q

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        x = x.permute(0, 2, 1).view
        
        (N, x.shape[2], H, W)

        out = self.mixffn(x)

        out = out + x

        out = self.proj(out)

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = myLayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class myLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        n, c, h, w = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # shape = [*, width, grid ** 2]
        x = x.reshape(n, c, h, w)
        return x

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale
    
class CformerBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        heads: int,
        output_dim: int,
        cross_dim: int,
        in_channels: int,
        input_resolution: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(cross_dim, output_dim, kernel_size=1)
        self.ff1 = FeedForward(embed_dims=in_channels, feedforward_channels=in_channels * 4)
        self.attn = CrossAttentionLayer(in_channels=in_channels,
                                        output_dim=in_channels,
                                        cross_dim=cross_dim,
                                        layers=layers,
                                        heads=heads,
                                        input_resolution=input_resolution)
        self.ff2 = FeedForward(embed_dims=in_channels, feedforward_channels=in_channels * 4)

        self.ff1 = Scale(0.5, PreNorm(in_channels, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(in_channels, self.ff2))

        self.post_norm = myLayerNorm(in_channels)
        

    def forward(self, x, cond):
        out1 = self.conv_in(cond)
        x = self.ff1(x)
        x = self.attn(q=x, k=cond)
        # x = self.conv(x) + x
        x = self.ff2(x)
        x = self.post_norm(x)
        out = out1 + x
        
        return out

class ConvBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        heads: int,
        output_dim: int,
        cross_dim: int,
        in_channels: int,
        input_resolution: int,
    ):
        super().__init__()
        self.activate = nn.GELU()  # Using nn.GELU directly

        conv1 = nn.Conv2d(in_channels, output_dim*2, kernel_size=1)
        conv2 = nn.Conv2d(output_dim*2, output_dim*2, kernel_size=3, padding=1)
        conv3 = nn.Conv2d(output_dim*2, output_dim, kernel_size=1)
        self.layers = nn.Sequential(
            conv1, self.activate, conv2, self.activate, conv3
        )

    def forward(self, x):
        # Assuming x is of shape (batch_size, channels, height, width)
        out = self.layers(x)

        return out

class DformerBlock(nn.Module):
    def __init__(
        self,
        layers: int,
        heads: int,
        output_dim: int,
        cross_dim: int,
        in_channels: int,
        input_resolution: int,
    ):
        super().__init__()
        # cross_dim = cross_dim // 2
        self.conv = nn.Conv2d(cross_dim+in_channels, output_dim, kernel_size=1)

        self.attn = CrossAttentionLayer(in_channels=in_channels,
                                        output_dim=in_channels,
                                        cross_dim=cross_dim,
                                        layers=layers,
                                        heads=heads,
                                        input_resolution=input_resolution)
    def forward(self, x, cond, time):
        # version 1
        input = torch.cat([x, cond], dim=1)
        res = self.conv(input)
        out = self.attn(q=res, k=cond)
        return out

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
        )
    )

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()

    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None
    ):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding

        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)
