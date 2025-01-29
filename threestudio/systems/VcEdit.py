import copy
from dataclasses import dataclass, field

from tqdm import tqdm

import torch
import threestudio
import os

# from threestudio.utils.clip_metrics import ClipSimilarity
from threestudio.systems.GaussianEditor import GaussianEditor
import cv2
import numpy as np


@threestudio.register("gsedit-system-edit")
class VcEdit(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))

        self.bg_gaussian = None

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.origin_frames = self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            fg_mask = self.update_mask()
            self.bg_gaussian = copy.deepcopy(self.gaussian)
            self.bg_gaussian.prune_points(fg_mask)

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
        if len(self.cfg.dds_target_prompt_processor) > 0:
            self.dds_target_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_target_prompt_processor)
        if len(self.cfg.dds_source_prompt_processor) > 0:
            self.dds_source_prompt_processor = threestudio.find(
                self.cfg.prompt_processor_type
            )(self.cfg.dds_source_prompt_processor)
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.loss.lambda_dds > 0:
            self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(
                self.cfg.second_guidance
            )

    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)

        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]

        out = self(batch, local=self.cfg.local_edit)#, renderbackground=bg_color)
        images = out["comp_rgb"]

        loss = 0.0
        # nerf2nerf loss
        if self.cfg.loss.lambda_l1 > 0 or self.cfg.loss.lambda_p > 0:

            if self.global_step % self.cfg.per_editing_step == 0:
                print("Update guidance at global step: ", self.global_step)
                curr_frames = self.render_all_view_no_cache(gaussian=copy.deepcopy(self.gaussian), with_semantic=False)
                result = self.guidance(
                    curr_frames,
                    copy.deepcopy(self.gaussian),
                    self.trainer.datamodule.train_dataset.scene.cameras,
                    self.pipe,
                    self.view_list,
                    calling_idx=self.global_step // self.cfg.per_editing_step,
                )
                for idx, view_idx in enumerate(self.view_list):
                    self.edit_frames[view_idx] = result["edit_images"][idx][None]

                concat_edit_frames = result["edit_images"].permute(1, 0, 2, 3).flatten(1, 2).cpu().numpy()
                concat_edit_frames = (concat_edit_frames.clip(0.0, 1.0) * 255.0).astype(np.uint8)
                concat_edit_frames = cv2.cvtColor(concat_edit_frames, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    self.get_save_path(f'edit_images_{self.guidance.tgt_prompt}_{self.cfg.per_editing_step}_{self.global_step}.png'),
                    concat_edit_frames)

            gt_images = []
            for img_index, cur_index in enumerate(batch_index):
                gt_images.append(self.edit_frames[cur_index])
            gt_images = torch.concatenate(gt_images, dim=0)

            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # dds loss
        if self.cfg.loss.lambda_dds > 0:
            dds_target_prompt_utils = self.dds_target_prompt_processor()
            dds_source_prompt_utils = self.dds_source_prompt_processor()

            second_guidance_out = self.second_guidance(
                out["comp_rgb"],
                torch.concatenate(
                    [self.origin_frames[idx] for idx in batch_index], dim=0
                ),
                dds_target_prompt_utils,
                dds_source_prompt_utils,
            )
            for name, value in second_guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(self.origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target)
                total_cos += abs(cos_sim.item())
        print(self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target, total_cos / len(self.view_list))
        self.log("train/clip_sim", total_cos / len(self.view_list))

    def render_all_view_with_aug(self):
        self.bg_aug_renders = {}
        for id in tqdm(self.view_list):
            cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
            cur_batch = {
                "index": id,
                "camera": [cur_cam],
                "height": self.trainer.datamodule.train_dataset.height,
                "width": self.trainer.datamodule.train_dataset.width,
            }

            view_out = []
            for bg_color in self.bg_aug_colors:
                out = self(cur_batch, renderbackground=bg_color)["comp_rgb"]
                view_out.append(out[0][None])
            self.bg_aug_renders[id] = torch.cat(view_out, dim=0)

    def multi_view_sync_attn(self, view_list, name):
        processors = self.guidance.pipe.unet.attn_processors
        for _, processor in processors.items():
            all_states = [torch.stack(processor.state[name][i], dim=0) for i in range(len(processor.state[name]))]
            all_states = torch.stack(all_states, dim=0).permute(1, 0, 2, 3, 4)
            assert len(all_states[0]) == len(view_list)
            all_states = all_states.view(-1, -1, -1, 64, 64, -1)