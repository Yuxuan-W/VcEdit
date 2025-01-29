import copy

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from .pipeline_ead import EditPipeline, ddcm_sampler
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from .ptp_utils import register_attention_control
from gaussiansplatting.gaussian_renderer import render
from threestudio.utils.perceptual import PerceptualLoss


class EditConsistPipeline(EditPipeline):
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            source_prompt: Union[str, List[str]],
            negative_prompt: Union[str, List[str]] = None,
            positive_prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            gaussian=None,
            cameras=None,
            render_pipe=None,
            view_list=None,
            attn_ctrl_steps=1,
            pred_ctrl_steps=5,
            blend_ctrl_steps=1,
            minigs_epochs=6,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 50,
            original_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            source_guidance_scale: Optional[float] = 1,
            num_images_per_prompt: Optional[int] = 1,
            eta: Optional[float] = 1.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            controllers=None,
            callbacks= None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            denoise_model: Optional[bool] = True,
    ):
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds_tuple, prompt_token_len = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        source_prompt_embeds_tuple, source_prompt_token_len = self.encode_prompt(
            source_prompt, device, num_images_per_prompt, do_classifier_free_guidance, positive_prompt, None
        )
        if prompt_embeds_tuple[1] is not None:
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])
        else:
            prompt_embeds = prompt_embeds_tuple[0]
        if source_prompt_embeds_tuple[1] is not None:
            source_prompt_embeds = torch.cat([source_prompt_embeds_tuple[1], source_prompt_embeds_tuple[0]])
        else:
            source_prompt_embeds = source_prompt_embeds_tuple[0]

        # 4. Get point colors and Preprocess image
        perceptual_loss = PerceptualLoss().eval().to(device)
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            original_inference_steps=original_inference_steps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents, clean_latents = self.prepare_latents_batches(
            image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, denoise_model,
            generator
        )
        if latents.shape[0] > 1:
            latents = latents[0][None].repeat(latents.shape[0], 1, 1, 1)

        source_latents = latents
        mutual_latents = latents

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        generator = extra_step_kwargs.pop("generator", None)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, t in enumerate(timesteps):

                noise = torch.randn(
                    latents[:1].shape,
                    dtype=latents.dtype, device=latents.device, generator=generator
                )

                # 8.1. cross attention consistency control
                if (step_idx + 1) % attn_ctrl_steps == 0:
                    print(f'Cross-Attention Consistency Control at t = {t} ...')
                    toy_controllers = [copy.deepcopy(controller) for controller in controllers]
                    for idx in range(len(latents)):

                        # get latents
                        latent = latents[idx][None]
                        source_latent = source_latents[idx][None]
                        mutual_latent = mutual_latents[idx][None]

                        # register the controller
                        controller = toy_controllers[idx]
                        register_attention_control(self, controller)

                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
                        source_latent_model_input = (
                            torch.cat([source_latent] * 2) if do_classifier_free_guidance else source_latent
                        )
                        mutual_latent_model_input = (
                            torch.cat([mutual_latent] * 2) if do_classifier_free_guidance else mutual_latent
                        )
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                        source_latent_model_input = self.scheduler.scale_model_input(source_latent_model_input, t)
                        mutual_latent_model_input = self.scheduler.scale_model_input(mutual_latent_model_input, t)

                        # predict the noise residual
                        assert do_classifier_free_guidance
                        concat_latent_model_input = torch.stack(  # source_latent is same initialized as tgt_latent
                            [
                                source_latent_model_input[0],
                                latent_model_input[0],
                                mutual_latent_model_input[0],  # mutual latents, the latent of the middle branch
                                source_latent_model_input[1],
                                latent_model_input[1],
                                mutual_latent_model_input[1],  # mutual latents, which plays as a middle results
                            ],
                            dim=0,
                        )  # 0 is unconditional latent, 1 is conditional latent, 0/1 is same initialized
                        concat_prompt_embeds = torch.stack(
                            [
                                source_prompt_embeds[0],
                                prompt_embeds[0],
                                source_prompt_embeds[0],
                                source_prompt_embeds[1],
                                prompt_embeds[1],
                                source_prompt_embeds[1],
                            ],
                            dim=0,
                        )  # 0 is unconditional prompt, 1 is conditional prompt

                        _ = self.unet(
                            concat_latent_model_input,
                            t,
                            cross_attention_kwargs=cross_attention_kwargs,
                            encoder_hidden_states=concat_prompt_embeds,
                        ).sample

                    # project and average the attn maps
                    attn_len = max(prompt_token_len, source_prompt_token_len)
                    attn_dtype = concat_latent_model_input.dtype
                    for module_name, attn_list in toy_controllers[0].consist_store.items():
                        if len(attn_list) == 0:
                            continue
                        for attn_idx in tqdm(range(len(attn_list))):
                            all_view_attn = []
                            for idx in range(len(view_list)):
                                view_attn = toy_controllers[idx].consist_store[module_name][attn_idx][:, :, :attn_len]
                                attn_size = int(view_attn.shape[1] ** 0.5)
                                view_attn = view_attn.reshape(-1, attn_size, attn_size, attn_len).permute(0, 3, 1, 2)
                                view_attn = F.interpolate(view_attn, size=(512, 512), mode="bilinear", align_corners=False)
                                view_attn = view_attn.flatten(0, 1).to(torch.float32)
                                all_view_attn.append(view_attn)

                            point_attn = []
                            point_attn_cnt = []
                            for c_idx in range(view_attn.shape[0]):
                                weight = torch.zeros((gaussian.get_opacity.shape[0], 1), dtype=torch.float32,
                                                     device="cuda")
                                weight_cnt = torch.zeros((gaussian.get_opacity.shape[0],), dtype=torch.int,
                                                         device="cuda")
                                for idx, view_idx in enumerate(view_list):
                                    gaussian.apply_weights(cameras[view_idx], weight, weight_cnt,
                                                           all_view_attn[idx][c_idx][None])
                                point_attn.append(weight)
                                point_attn_cnt.append(weight_cnt)
                            point_attn = torch.cat(point_attn, dim=-1)
                            point_attn_cnt = torch.stack(point_attn_cnt, dim=-1)
                            point_attn = point_attn / (point_attn_cnt + 1e-8)

                            del all_view_attn
                            torch.cuda.empty_cache()

                            # rendering
                            con_view_attn = []
                            assert point_attn.shape[1] % 3 == 0
                            for c_idx in range(point_attn.shape[1] // 3):
                                point_attn_ = point_attn[:, c_idx * 3:(c_idx + 1) * 3]
                                con_view_attn_ = []
                                for idx, view_idx in enumerate(view_list):
                                    con_view_attn_.append(render(cameras[view_idx], gaussian, render_pipe,
                                                                 override_color=point_attn_,
                                                                 bg_color=torch.tensor([0, 0, 0], dtype=torch.float32,
                                                                                       device="cuda"))['render'])
                                con_view_attn.append(torch.stack(con_view_attn_, dim=0))
                            con_view_attn = torch.cat(con_view_attn, dim=1)

                            # write to controllers
                            con_view_attn = F.interpolate(con_view_attn, size=(attn_size, attn_size),
                                                          mode="bilinear", align_corners=False)
                            con_view_attn = con_view_attn.reshape(len(view_list), -1, attn_len, attn_size, attn_size
                                                                  ).permute(0, 1, 3, 4, 2).flatten(2, 3)
                            for idx, view_idx in enumerate(view_list):
                                toy_controllers[idx].consist_store[module_name][attn_idx][:, :, :attn_len] = con_view_attn[idx]
                                controllers[idx].consist_store[module_name].append(copy.deepcopy(
                                    toy_controllers[idx].consist_store[module_name][attn_idx].to(attn_dtype).clamp(0, 1)))

                            del con_view_attn, point_attn, point_attn_cnt
                            torch.cuda.empty_cache()
                    for controller in controllers:
                        controller.enable_consist()
                    del toy_controllers
                    torch.cuda.empty_cache()

                # 8.2. forward each view
                view_preds = []
                for idx in range(len(latents)):

                    # get latents
                    latent = latents[idx][None]
                    clean_latent = clean_latents[idx][None]
                    source_latent = source_latents[idx][None]
                    mutual_latent = mutual_latents[idx][None]

                    # register the controller
                    controller = controllers[idx]
                    register_attention_control(self, controller)

                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
                    source_latent_model_input = (
                        torch.cat([source_latent] * 2) if do_classifier_free_guidance else source_latent
                    )
                    mutual_latent_model_input = (
                        torch.cat([mutual_latent] * 2) if do_classifier_free_guidance else mutual_latent
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    source_latent_model_input = self.scheduler.scale_model_input(source_latent_model_input, t)
                    mutual_latent_model_input = self.scheduler.scale_model_input(mutual_latent_model_input, t)

                    # predict the noise residual
                    assert do_classifier_free_guidance
                    concat_latent_model_input = torch.stack(  # source_latent is same initialized as tgt_latent
                        [
                            source_latent_model_input[0],
                            latent_model_input[0],
                            mutual_latent_model_input[0],  # mutual latents, the latent of the middle branch
                            source_latent_model_input[1],
                            latent_model_input[1],
                            mutual_latent_model_input[1],  # mutual latents, which plays as a middle results
                        ],
                        dim=0,
                    )  # 0 is unconditional latent, 1 is conditional latent, 0/1 is same initialized
                    concat_prompt_embeds = torch.stack(
                        [
                            source_prompt_embeds[0],
                            prompt_embeds[0],
                            source_prompt_embeds[0],
                            source_prompt_embeds[1],
                            prompt_embeds[1],
                            source_prompt_embeds[1],
                        ],
                        dim=0,
                    )  # 0 is unconditional prompt, 1 is conditional prompt

                    concat_noise_pred = self.unet(
                        concat_latent_model_input,
                        t,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_hidden_states=concat_prompt_embeds,
                    ).sample

                    # perform guidance
                    (source_noise_pred_uncond, noise_pred_uncond,
                     mutual_noise_pred_uncond, source_noise_pred_text,
                     noise_pred_text, mutual_noise_pred_text) = concat_noise_pred.chunk(6, dim=0)

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    source_noise_pred = source_noise_pred_uncond + source_guidance_scale * (
                            source_noise_pred_text - source_noise_pred_uncond
                    )
                    mutual_noise_pred = mutual_noise_pred_uncond + source_guidance_scale * (
                            mutual_noise_pred_text - mutual_noise_pred_uncond
                    )

                    # all the latents here are not used for computing
                    _, latent, pred_x0, _ = ddcm_sampler(
                        self.scheduler, source_latent,
                        latent, t,
                        source_noise_pred, noise_pred,
                        clean_latent, noise=noise,
                        eta=eta, to_next=False,
                        **extra_step_kwargs
                    )

                    # all the latents here are not used for computing
                    source_latent, mutual_latent, pred_mu0, alpha_prod_t_prev = ddcm_sampler(
                        self.scheduler, source_latent,
                        mutual_latent, t,
                        source_noise_pred, mutual_noise_pred,
                        clean_latent, noise=noise,
                        eta=eta, to_next=False,
                        **extra_step_kwargs
                    )

                    view_preds.append(torch.cat([pred_x0, pred_mu0], dim=1))
                view_preds = torch.cat(view_preds, dim=0)

                for controller in controllers:
                    controller.reset_consist()
                    controller.disable_consist()

                # 8.3. prediction consistency control
                # ================== consist control for pred_x0 and pred_mu0 ==================
                if (step_idx + 1) % pred_ctrl_steps == 0:

                    # inverse project images to 3d
                    print(f'Prediction Consistency Control at t = {t} ...')
                    print(f'--step 1: inverse project images to 3d ...')
                    view_img_preds = torch.cat([self.decode_batches(view_preds[:, :4]),
                                                self.decode_batches(view_preds[:, 4:])], dim=1)
                    view_img_preds = (view_img_preds / 2 + 0.5).clamp(0, 1).to(torch.float32)

                    gaussian_color_x0 = self.mini_gaussian_training(copy.deepcopy(gaussian), view_list,
                                                                 view_img_preds[:, :3],  cameras,
                                                                 render_pipe, perceptual_loss, minigs_epochs)
                    gaussian_color_mu0 = self.mini_gaussian_training(copy.deepcopy(gaussian), view_list,
                                                                  view_img_preds[:, 3:],  cameras,
                                                                  render_pipe, perceptual_loss, minigs_epochs)
                    # point_color = [point_color_x0, point_color_mu0]

                    # render back to images
                    print(f'--step 2: render back to images ...')
                    con_view_preds_x0 = []
                    con_view_preds_mu0 = []
                    for idx, view_idx in enumerate(view_list):
                        # gaussian._features_dc = point_color[0]
                        x0_render_pkg = render(cameras[view_idx], gaussian_color_x0, render_pipe,
                                               bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
                        # gaussian._features_dc = point_color[1]
                        mu0_render_pkg = render(cameras[view_idx], gaussian_color_mu0, render_pipe,
                                                bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))

                        con_view_preds_x0.append(x0_render_pkg['render'])
                        con_view_preds_mu0.append(mu0_render_pkg['render'])
                        # import matplotlib.pyplot as plt
                        # plt.imshow(render_pkg['render'].detach().cpu().numpy().transpose(1, 2, 0))
                        # plt.show()
                    con_view_preds_x0 = torch.stack(con_view_preds_x0, dim=0)
                    con_view_preds_mu0 = torch.stack(con_view_preds_mu0, dim=0)

                    # re-encode images to latents
                    con_view_preds_x0 = (con_view_preds_x0 * 2 - 1).to(torch.float32).clamp(-1, 1)
                    con_view_preds_mu0 = (con_view_preds_mu0 * 2 - 1).to(torch.float32).clamp(-1, 1)
                    _, con_view_preds_x0 = self.prepare_latents_batches(
                        con_view_preds_x0, latent_timestep, batch_size, num_images_per_prompt,
                        prompt_embeds.dtype, device, denoise_model, generator
                    )
                    _, con_view_preds_mu0 = self.prepare_latents_batches(
                        con_view_preds_mu0, latent_timestep, batch_size, num_images_per_prompt,
                        prompt_embeds.dtype, device, denoise_model, generator
                    )
                    view_preds = torch.cat([con_view_preds_x0, con_view_preds_mu0], dim=1)

                    del gaussian_color_x0, gaussian_color_mu0, con_view_preds_x0, con_view_preds_mu0
                    torch.cuda.empty_cache()

                # 8.4. blending map consistency control
                # ================== consist control for blend maps ==================
                maps = []
                for idx, view_idx in enumerate(view_list):
                    maps.append(controllers[idx].local_blend.get_blend_maps(controllers[idx].attention_store,
                                                                            view_preds.shape[-2], view_preds.shape[-1]))
                maps = torch.cat(maps, dim=0)

                if (step_idx + 1) % blend_ctrl_steps == 0:
                    print(f'Blend Cross-Attention Consistency Control at t = {t} ...')
                    print(f'--step 1: inverse project maps to 3d ...')
                    maps = F.interpolate(maps, size=(512, 512))
                    point_maps = []
                    point_maps_cnt = []
                    for c_idx in range(maps.shape[1]):
                        weight = torch.zeros((gaussian.get_opacity.shape[0], 1), dtype=torch.float32, device="cuda")
                        weight_cnt = torch.zeros((gaussian.get_opacity.shape[0],), dtype=torch.int, device="cuda")
                        for idx, view_idx in enumerate(view_list):
                            gaussian.apply_weights(cameras[view_idx], weight, weight_cnt, maps[idx][c_idx][None])
                        point_maps.append(weight)
                        point_maps_cnt.append(weight_cnt)
                    point_maps = torch.cat(point_maps, dim=-1)
                    point_maps_cnt = torch.stack(point_maps_cnt, dim=-1)
                    point_maps = point_maps / (point_maps_cnt + 1e-8)

                    print(f'--step 2: render back to maps ...')
                    maps = []
                    for idx, view_idx in enumerate(view_list):
                        maps.append(render(cameras[view_idx], gaussian, render_pipe, override_color=point_maps,
                                           bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))['render'])

                    maps = F.interpolate(torch.stack(maps, dim=0), size=(view_preds.shape[-2], view_preds.shape[-1]))

                # 8.5. update latents and call callback
                # ================== update latents and call callback ==================
                print(f'Update Latents & Callback at t = {t} ...')
                latents = []
                mutual_latents = []
                source_latents = []

                for idx, view_idx in enumerate(view_list):
                    callback = callbacks[idx] if callbacks is not None else None
                    map = maps[idx][None] if callbacks is not None else None

                    pred_x0 = view_preds[idx][None][:, :4]
                    pred_mu0 = view_preds[idx][None][:, 4:]
                    x0 = clean_latents[idx][None]

                    latent = alpha_prod_t_prev ** (0.5) * pred_x0 + (1 - alpha_prod_t_prev) ** 0.5 * noise
                    mutual_latent = alpha_prod_t_prev ** (0.5) * pred_mu0 + (1 - alpha_prod_t_prev) ** 0.5 * noise
                    source_latent = alpha_prod_t_prev ** (0.5) * x0 + (1 - alpha_prod_t_prev) ** 0.5 * noise

                    # call the callback, if provided
                    if step_idx == len(timesteps) - 1 or ((step_idx + 1) > num_warmup_steps
                                                          and (step_idx + 1) % self.scheduler.order == 0):
                        if callback is not None and step_idx % callback_steps == 0:
                            alpha_prod_t = self.scheduler.alphas_cumprod[t]
                            mutual_latent, latent = callback(step_idx, t, source_latent, latent, mutual_latent,
                                                             alpha_prod_t, maps=map)

                    latents.append(latent)
                    mutual_latents.append(mutual_latent)
                    source_latents.append(source_latent)

                latents = torch.cat(latents, dim=0)
                mutual_latents = torch.cat(mutual_latents, dim=0)
                source_latents = torch.cat(source_latents, dim=0)
                all_pred_x0 = view_preds[:, :4]

                print(f'Finished t = {t}.')
                self.scheduler._step_index += 1
                progress_bar.update()

                del maps, view_preds
                torch.cuda.empty_cache()

        # 9. Post-processing
        if not output_type == "latent":
            image = self.decode_batches(all_pred_x0)
        else:
            image = pred_x0

        do_denormalize = [True] * image.shape[0]
        nsfw_content_detected = [False] * image.shape[0]

        image = torch.stack(
            [self.image_processor.denormalize(image[i]) if do_denormalize[i] else image[i] for i in
             range(image.shape[0])]
        )

        if not return_dict:
            return image

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=nsfw_content_detected)

    def prepare_latents_batches(self, image, latent_timestep, batch_size, num_images_per_prompt, dtype,
                                device, denoise_model, generator, max_images_per_batch=10):
        all_latents = []
        all_clean_latents = []

        # Calculate the number of batches needed
        num_batches = (image.shape[0] + max_images_per_batch - 1) // max_images_per_batch

        for i in range(num_batches):
            # Calculate start and end indices for the current batch
            start_idx = i * max_images_per_batch
            end_idx = min(start_idx + max_images_per_batch, image.shape[0])

            # Slice the image tensor to get the current batch
            image_batch = image[start_idx:end_idx]

            # Call self.prepare_latents() for the current batch
            latents, clean_latents = self.prepare_latents(
                image_batch, latent_timestep, batch_size, num_images_per_prompt, dtype, device,
                denoise_model, generator
            )

            # Append the resulting latents and clean_latents to their respective lists
            all_latents.append(latents)
            all_clean_latents.append(clean_latents)

        # Concatenate all latents and clean_latents along dimension 0
        all_latents = torch.cat(all_latents, dim=0)
        all_clean_latents = torch.cat(all_clean_latents, dim=0)

        return all_latents, all_clean_latents

    def decode_batches(self, pred_x0, max_images_per_batch=5):
        num_batches = (pred_x0.shape[0] + max_images_per_batch - 1) // max_images_per_batch
        images = []
        for i in range(num_batches):
            # Calculate start and end indices for the current batch
            start_idx = i * max_images_per_batch
            end_idx = min(start_idx + max_images_per_batch, pred_x0.shape[0])

            # Slice the image tensor to get the current batch
            pred_x0_batch = pred_x0[start_idx:end_idx]

            image = self.vae.decode(pred_x0_batch / self.vae.config.scaling_factor, return_dict=False)[0]
            images.append(image)
        images = torch.cat(images, dim=0)
        return images

    def mini_gaussian_training(self, gaussian, view_list, images, cameras, render_pipe, perceptual_loss, minigs_epochs):

        # build a random train_loader for images in shape [batch_size, 3, 512, 512]
        class RandomDataset(Dataset):
            def __init__(self, images):
                self.images = images

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return idx, self.images[idx]
        train_loader = DataLoader(RandomDataset(images), batch_size=1, shuffle=True, num_workers=0)

        gaussian._xyz.requires_grad = True
        gaussian._features_dc.requires_grad = True
        gaussian._features_rest.requires_grad = True
        gaussian._opacity.requires_grad = True
        gaussian._scaling.requires_grad = True
        gaussian._rotation.requires_grad = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        # optimizer = torch.optim.Adam([{"params": [gaussian._features_dc], "lr": 0.0625, "name": "f_dc"}],
        #                              lr=0.0, betas=(0.9, 0.99), eps=1e-15)
        optimizer = gaussian.optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        num_epochs = minigs_epochs
        with torch.enable_grad():
            for epoch in range(num_epochs):
                epoch_avg_loss = 0.0
                for i, (idx, gt_images) in enumerate(train_loader):
                    view_idx = view_list[idx]
                    gt_image = gt_images[0].to(device)

                    render_image = render(cameras[view_idx], gaussian, render_pipe, bg_color=bg_color)['render']

                    # mse_loss = F.mse_loss(render_image, gt_image)
                    l1_loss = F.l1_loss(render_image, gt_image)
                    p_loss = perceptual_loss(render_image, gt_image).sum()
                    loss = 10 * l1_loss + 10 * p_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_avg_loss += loss.item()

                    # if epoch == num_epochs - 1 and view_idx < 4:
                    #     import matplotlib.pyplot as plt
                    #     plt.imshow(render_image.detach().to(torch.float32).permute(1, 2, 0).cpu().numpy())
                    #     plt.show()
                # print(f'Epoch {epoch}, Loss {epoch_avg_loss / len(train_loader)}')
                scheduler.step()

        gaussian._features_dc.requires_grad = False
        point_color = copy.deepcopy(gaussian._features_dc.detach())
        torch.cuda.empty_cache()
        return gaussian