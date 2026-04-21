import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import List, Optional, Tuple, Union

from diffusers.utils import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from src.models.clip import DformerBlock


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class InteractNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples_q (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples_v: Tuple[torch.Tensor]
    mid_block_res_sample_v: torch.Tensor
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class InteractNet(ModelMixin, ConfigMixin):
    """

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        conditioning_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        in_channels: Tuple[int, ...] = (
            320,
            320,
            320,
            320,
            640,
            640,
            640,
            1280,
            1280,
            1280,
            1280,
            1280,
            1280,
        ),
        block_out_resolutions: Tuple[int, ...] = (
            64,
            64,
            64,
            32,
            32,
            32,
            16,
            16,
            16,
            8,
            8,
            8,
            8,
        ),
        layers_per_block: int = 2,
        attn_layers: int = 1,
        attn_heads: int = 8,
    ):
        super().__init__()
        q_in_channels = in_channels
        kv_in_channels = in_channels
        block_out_channels = in_channels

        self.attn_blocks_down = nn.ModuleList([])
        self.conv_blocks_down = nn.ModuleList([])
        self.attn_blocks_mid = None
        self.conv_blocks_mid = None

        for i, in_channel in enumerate(in_channels):
            output_channel = block_out_channels[i]
            kv_in_channel = kv_in_channels[i] * 2
            q_in_channel = q_in_channels[i]
            input_resolution = block_out_resolutions[i]

            is_final_block = i == len(in_channels) - 1

            if not is_final_block:
                interactnet_attn_block = DformerBlock(
                    in_channels=q_in_channel,
                    cross_dim=kv_in_channel,
                    output_dim=output_channel,
                    layers=attn_layers,
                    heads=attn_heads,
                    input_resolution=input_resolution,
                )
                interactnet_attn_block = self.zero_module(interactnet_attn_block)
                self.attn_blocks_down.append(interactnet_attn_block)
            else:
                self.attn_blocks_mid = DformerBlock(
                    in_channels=q_in_channel,
                    cross_dim=kv_in_channel,
                    output_dim=output_channel,
                    layers=attn_layers,
                    heads=attn_heads,
                    input_resolution=input_resolution,
                )
                self.attn_blocks_mid = self.zero_module(self.attn_blocks_mid)

    def zero_module(self, module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    def uniform_module(self, module, a=-0.1, b=0.1):
        for p in module.parameters():
            nn.init.uniform_(p, a, b)
        return module

    def forward(
        self,
        temb=None,
        timesteps=None,
        down_block_res_samples_noise: Optional[Tuple[torch.Tensor]] = None,
        down_block_res_samples_rough: Optional[Tuple[torch.Tensor]] = None,
        down_block_res_samples_image: Optional[Tuple[torch.Tensor]] = None,
        mid_block_sample_noise: Optional[torch.Tensor] = None,
        mid_block_sample_rough: Optional[torch.Tensor] = None,
        mid_block_sample_image: Optional[torch.Tensor] = None,
        guess_mode: bool = False,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[
        InteractNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]
    ]:
        """Compute interaction features for the generative UNet."""
        down_block_res_samples = ()

        for (
            down_block_res_sample,
            res_sample_image,
            res_sample_rough,
            attn_block,
        ) in zip(
            down_block_res_samples_noise,
            down_block_res_samples_image,
            down_block_res_samples_rough,
            self.attn_blocks_down,
        ):
            down_res_sample_cond = torch.cat(
                [res_sample_image, res_sample_rough], dim=1
            )
            down_block_res_sample = attn_block(
                down_block_res_sample, down_res_sample_cond, timesteps
            )
            down_block_res_samples = down_block_res_samples + (down_block_res_sample,)

        mid_res_sample_cond = torch.cat(
            [mid_block_sample_image, mid_block_sample_rough], dim=1
        )
        mid_block_res_sample = self.attn_blocks_mid(
            mid_block_sample_noise, mid_res_sample_cond, timesteps
        )
        down_block_res_samples_v = None
        mid_block_res_sample_v = None

        if not return_dict:
            return (
                down_block_res_samples_v,
                mid_block_res_sample_v,
                down_block_res_samples,
                mid_block_res_sample,
            )

        return InteractNetOutput(
            down_block_res_samples_v=down_block_res_samples_v,
            mid_block_res_sample_v=mid_block_res_sample_v,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )


class SIDModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        unet_gen,
        interact_net,
        label_embed_net,
        unet_ori,
        z_dims: Optional[List[int]] = None,
        projector_dim: int = 2048,
        spatial_in_tokens: int = 64,
        spatial_out_tokens: int = 1024,
    ):
        super().__init__()

        self.unet_gen = unet_gen
        self.interact_net = interact_net
        self.label_embed_net = label_embed_net
        self.unet_ori = unet_ori
        self.spatial_in_tokens = spatial_in_tokens
        self.spatial_out_tokens = spatial_out_tokens

        self.projectors = None
        if z_dims:
            if hasattr(unet_gen, "config") and hasattr(
                unet_gen.config, "block_out_channels"
            ):
                inner_dim = unet_gen.config.block_out_channels[-1]
            else:
                inner_dim = unet_gen.conv_out.in_channels
            self.projectors = nn.ModuleList(
                [build_mlp(inner_dim, projector_dim, z_dim) for z_dim in z_dims]
            )
            self._initialize_projectors()

    def _initialize_projectors(self):
        if self.projectors is None:
            return

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for projector in self.projectors:
            projector.apply(_basic_init)

    def _build_representators(
        self,
        down_block_res_samples_q,
        down_block_res_samples,
        target_token_grid: Optional[Tuple[int, int]] = None,
    ):
        if self.projectors is None:
            return None
        x = down_block_res_samples[-1]
        bsz, channels, height, width = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        representators_tilde = []
        for projector in self.projectors:
            z = projector(x.reshape(-1, channels)).view(bsz, height, width, -1)
            z = z.permute(0, 3, 1, 2)
            if target_token_grid is not None and tuple(target_token_grid) != (height, width):
                z = F.interpolate(
                    z,
                    size=target_token_grid,
                    mode="bilinear",
                    align_corners=False,
                )
            representators_tilde.append(z.flatten(2).permute(0, 2, 1))
        return representators_tilde

    def connect_process(
        self, noise_latents, img_latents, seg_latents, timesteps, encoder_hidden_states
    ):
        with torch.no_grad():
            down_block_res_samples_kv_1, mid_block_sample_kv_1 = (
                self.unet_ori.get_blocks(
                    img_latents,
                    timestep=1,
                    encoder_hidden_states=encoder_hidden_states,
                )
            )

            down_block_res_samples_kv_2, mid_block_sample_kv_2 = (
                self.unet_ori.get_blocks(
                    seg_latents,
                    timestep=1,
                    encoder_hidden_states=encoder_hidden_states,
                )
            )

        down_block_res_samples_q, mid_block_sample_q, emb_q = self.unet_gen.get_blocks(
            noise_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
        interact_output_1 = self.interact_net(
            temb=emb_q,
            timesteps=timesteps,
            down_block_res_samples_noise=down_block_res_samples_q,
            down_block_res_samples_rough=down_block_res_samples_kv_2,
            down_block_res_samples_image=down_block_res_samples_kv_1,
            mid_block_sample_noise=mid_block_sample_q,
            mid_block_sample_rough=mid_block_sample_kv_2,
            mid_block_sample_image=mid_block_sample_kv_1,
        )
        down_block_res_samples = interact_output_1.down_block_res_samples
        mid_block_sample = interact_output_1.mid_block_res_sample

        return down_block_res_samples, mid_block_sample, down_block_res_samples_q

    def get_gen_pred(
        self,
        noise_latents,
        cond_latents_1,
        cond_latents_2,
        timesteps,
        encoder_hidden_states,
        target_token_grid: Optional[Tuple[int, int]] = None,
    ):
        down_block_res_samples, mid_block_sample, _ = self.connect_process(
            noise_latents,
            cond_latents_1,
            cond_latents_2,
            timesteps,
            encoder_hidden_states,
        )
        representators_tilde = self._build_representators(
            None,
            down_block_res_samples,
            target_token_grid=target_token_grid,
        )

        unet_output = self.unet_gen(
            noise_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_sample,
        )
        if isinstance(unet_output, tuple):
            seg_model_pred, representators_tilde = unet_output
        else:
            seg_model_pred = unet_output
        seg_model_pred = seg_model_pred.sample
        return seg_model_pred, representators_tilde

    def forward(
        self,
        noise_latents=None,
        cond_latents_1=None,
        cond_latents_2=None,
        timesteps=None,
        encoder_hidden_states=None,
        target_token_grid: Optional[Tuple[int, int]] = None,
        return_representations: bool = False,
    ):
        seg_model_pred, representators_tilde = self.get_gen_pred(
            noise_latents,
            cond_latents_1,
            cond_latents_2,
            timesteps,
            encoder_hidden_states,
            target_token_grid=target_token_grid,
        )
        if return_representations:
            return seg_model_pred, representators_tilde
        return seg_model_pred

    @torch.no_grad()
    def predict(
        self,
        noise_latents=None,
        img_latents=None,
        seg_latents=None,
        timesteps=None,
        encoder_hidden_states=None,
    ):

        down_block_res_samples, mid_block_sample, _ = self.connect_process(
            noise_latents.chunk(2)[1],
            img_latents,
            seg_latents,
            timesteps,
            encoder_hidden_states.chunk(2)[1],
        )
        representators_tilde = self._build_representators(None, down_block_res_samples)

        down_block_res_samples = [
            torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples
        ]
        mid_block_sample = torch.cat(
            [torch.zeros_like(mid_block_sample), mid_block_sample]
        )

        unet_output = self.unet_gen(
            noise_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_sample,
        )
        if isinstance(unet_output, tuple):
            model_pred, representators_tilde = unet_output
        else:
            model_pred = unet_output
        model_pred = model_pred.sample
        return model_pred, representators_tilde


class LabelEmbedNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_classes: int = 6,
        label_emb_inchannels: int = 3,
        bit_scale: float = 1,
    ):
        super().__init__()
        self.bit_scale = bit_scale

        self.embedding_table = nn.Embedding(num_classes, label_emb_inchannels)
        self.embedding_conv = nn.Sequential(
            nn.Conv2d(label_emb_inchannels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def encode(self, label_down):
        label_down = self.embedding_table(label_down).squeeze(1).permute(0, 3, 1, 2)
        label_down = torch.sigmoid(label_down) * 2 - 1
        return label_down

    def decode(self, embedding):
        logits = self.embedding_conv(embedding)
        return logits
