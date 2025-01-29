from dataclasses import dataclass
import cv2
import os
import numpy as np
import torch
import copy
from diffusers import DDIMScheduler, LCMScheduler
from diffusers.utils.import_utils import is_xformers_available
from infedit.pipeline_con import EditConsistPipeline
from infedit.infedit import LocalBlend, AttentionRefine
import infedit.ptp_utils as ptp_utils

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-infedit-guidance")
class InfEditGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        model_name_or_path: str = "SimianLuo/LCM_Dreamshaper_v7"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 15
        attn_ctrl_steps: int = 7
        pred_ctrl_steps: int = 5
        blend_ctrl_steps: int = 1
        minigs_epochs: int = 6

        skip_first_ctrl: bool = False

        use_sds: bool = False

        src_prompt: str = ""
        tgt_prompt: str = ""
        cross_attn_th: float = 0.7
        self_attn_th: float = 0.7
        src_blend_th: float = 0.5
        tgt_blend_th: float = 0.5
        guidance_scale: float = 2.0

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading InfEdit ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        self.scheduler = LCMScheduler.from_config(self.cfg.model_name_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                                  subfolder="scheduler", cache_dir=self.cfg.cache_dir)
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        self.pipe = EditConsistPipeline.from_pretrained(self.cfg.model_name_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                                 scheduler=self.scheduler, torch_dtype=torch.float16).to(self.device)

        self.src_prompt = self.cfg.src_prompt
        self.tgt_prompt = self.cfg.tgt_prompt

        # create the CAC controller.
        local_blend = LocalBlend(thresh_e=self.cfg.src_blend_th, thresh_m=self.cfg.tgt_blend_th, save_inter=False)
        self.controller = AttentionRefine([self.src_prompt, self.tgt_prompt], [[self.tgt_prompt, '']],
                                     self.pipe.tokenizer, self.pipe.text_encoder,
                                     num_steps=self.cfg.diffusion_steps,
                                     start_steps=0,
                                     cross_replace_steps=self.cfg.cross_attn_th,
                                     self_replace_steps=self.cfg.self_attn_th,
                                     local_blend=local_blend
                                     )

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded InfEdit!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def __call__(
        self,
        rgb, gaussian, cameras, render_pipe, view_list, calling_idx,
        **kwargs,
    ):
        batch_size = len(rgb)
        _, H, W, _ = rgb[0].shape
        assert H == 512 and W == 512, "Image size must be 512x512"

        rgb = torch.cat([v.permute(0, 3, 1, 2) for _, v in rgb.items()], dim=0)
        # rgb = rgb.permute(0, 3, 1, 2)

        if calling_idx > 0:
            local_blend = LocalBlend(thresh_e=self.cfg.src_blend_th, thresh_m=self.cfg.tgt_blend_th, save_inter=False)
            self.controller = AttentionRefine([self.tgt_prompt, self.tgt_prompt], [[self.tgt_prompt, '']],
                                              self.pipe.tokenizer, self.pipe.text_encoder,
                                              num_steps=self.cfg.diffusion_steps,
                                              start_steps=0,
                                              cross_replace_steps=self.cfg.cross_attn_th,
                                              self_replace_steps=self.cfg.self_attn_th,
                                              local_blend=local_blend)

        torch.manual_seed(0)
        self.controller.reset()
        controllers = [copy.deepcopy(self.controller) for _ in range(batch_size)]

        skip_ctrl = self.cfg.skip_first_ctrl and calling_idx == 0

        # ptp_utils.register_attention_control(self.pipe, self.controller)
        results = self.pipe(prompt=self.tgt_prompt,
                       source_prompt=self.src_prompt if calling_idx == 0 else self.tgt_prompt,
                       positive_prompt='',
                       negative_prompt='',
                       image=rgb,
                        gaussian=gaussian,
                        cameras=cameras,
                        render_pipe=render_pipe,
                        view_list=view_list,
                        attn_ctrl_steps=self.cfg.attn_ctrl_steps if not skip_ctrl else 1000,
                        pred_ctrl_steps=self.cfg.pred_ctrl_steps if not skip_ctrl else 1000,
                        blend_ctrl_steps=self.cfg.blend_ctrl_steps,
                       num_inference_steps=self.cfg.diffusion_steps,
                       eta=1,
                       strength=1,
                       guidance_scale=self.cfg.guidance_scale,
                       source_guidance_scale=1,
                       denoise_model=False,
                        controllers=controllers,
                       callbacks=[controller.step_callback for controller in controllers]
                       )

        edit_images = results.images

        if self.cfg.use_sds:
            raise NotImplementedError("Haven't implemented yet.")

        return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def call_single_img(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        noise: Float[Tensor, "B 4 DH DW"] = None,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape

        assert H == 512 and W == 512, "Image size must be 512x512"
        rgb = rgb.permute(0, 3, 1, 2)

        torch.manual_seed(0)
        self.controller.reset()

        ptp_utils.register_attention_control(self.pipe, self.controller)
        results = self.pipe(prompt=self.tgt_prompt,
                       source_prompt=self.src_prompt,
                       positive_prompt='',
                       negative_prompt='',
                       image=rgb,
                       num_inference_steps=self.cfg.diffusion_steps,
                       eta=1,
                       strength=1,
                       guidance_scale=2,
                       source_guidance_scale=1,
                       denoise_model=False,
                       callback=self.controller.step_callback
                       )

        edit_images = results.images

        if self.cfg.use_sds:
            raise NotImplementedError("Haven't implemented yet.")

        return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/instructpix2pix.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )
    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (
            guidance_out["edit_images"][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .clip(0, 1)
            .numpy()
            * 255
        )
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    import os

    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
