from diffusers import LCMScheduler
import os
import torch
from PIL import Image
import torch.nn.functional as nnf
from typing import Optional, Union, Tuple, List, Callable, Dict
import abc
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

import infedit.ptp_utils as ptp_utils
import infedit.seq_aligner as seq_aligner
from infedit.pipeline_ead import EditPipeline


class LocalBlend:

    def get_mask(self, x_t, maps, word_idx, thresh, i):
        maps = maps * word_idx.reshape(1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, 1:self.len - 1]).mean(0, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask

    def save_image(self, mask, i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        # print(image.shape)
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        # print(image.shape)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"inter/{caption}"):
            os.mkdir(f"inter/{caption}")
        ptp_utils.save_images(image, f"inter/{caption}/{i}.jpg")

    def _get_mask_helper(self, shape, maps, word_idx):
        maps = maps * word_idx.reshape(1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, 1:self.len - 1]).mean(0, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(shape))
        maps = maps / (maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0] + 1e-6)
        return maps

    def get_blend_maps(self, attention_store, H, W):
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        h, w = ((H + 1) // 2 + 1) // 2, ((W + 1) // 2 + 1) // 2
        maps = [
            item.reshape(2, -1, 1, h // int((h * w / item.shape[-2]) ** 0.5), w // int((h * w / item.shape[-2]) ** 0.5),
                         77) for item in maps]
        maps = torch.cat(maps, dim=1)

        maps_s = maps[0, :]
        maps_m = maps[1, :]

        map_e = self._get_mask_helper((H, W), maps_m, self.alpha_e)
        map_m = self._get_mask_helper((H, W), maps_s, (self.alpha_m - self.alpha_me))
        map_me = self._get_mask_helper((H, W), maps_m, self.alpha_me)

        return torch.cat([map_e, map_m, map_me], dim=1)

    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False, maps=None):
        thresh_e = temperature / alpha_prod ** (0.5)
        thresh_m = self.thresh_m
        if thresh_e < self.thresh_e:
            thresh_e = self.thresh_e

        if maps is None:
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            h, w = x_t.shape[2], x_t.shape[3]
            h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
            maps = [
                item.reshape(2, -1, 1, h // int((h * w / item.shape[-2]) ** 0.5), w // int((h * w / item.shape[-2]) ** 0.5),
                             77) for item in maps]
            maps = torch.cat(maps, dim=1)

            maps_s = maps[0, :]
            maps_m = maps[1, :]
            mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
            mask_m = self.get_mask(x_t, maps_s, (self.alpha_m - self.alpha_me), thresh_m, i)
            mask_me = self.get_mask(x_t, maps_m, self.alpha_me, self.thresh_e, i)
        else:
            masks = []
            threshs = [thresh_e, thresh_m, thresh_e]
            for i in range(3):
                map = maps[:, None, i, :]
                map = map / map.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
                mask = map > threshs[i]
                masks.append(mask)
            mask_e, mask_m, mask_me = masks

        if self.save_inter:
            self.save_image(mask_e, i, "mask_e")
            self.save_image(mask_m, i, "mask_m")
            self.save_image(mask_me, i, "mask_me")

        if self.alpha_e.sum() == 0:
            x_t_out = x_t
        else:
            x_t_out = torch.where(mask_e, x_t, x_m)
        x_t_out = torch.where(mask_m, x_s, x_t_out)
        if use_xm:
            x_t_out = torch.where(mask_me, x_m, x_t_out)

        return x_m, x_t_out

    def __init__(self, thresh_e=0.3, thresh_m=0.3, save_inter=False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter

    def set_map(self, ms, alpha, alpha_e, alpha_m, len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)    # 只对后半部分，即conditional处理
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers: # 除以2是因为self和cross
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def self_attn_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, consist_store=None):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
            if consist_store is not None:
                self.consist_store[key].append(consist_store)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.consist_store = self.get_empty_store()
        self.attention_store = {}

    def reset_consist(self):
        self.consist_store = self.get_empty_store()
        self.consist_cnt = {'down': 0, 'mid': 0, 'up': 0}

    def enable_consist(self):
        self.use_consist = True

    def disable_consist(self):
        self.use_consist = False

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.consist_store = self.get_empty_store()
        self.use_consist = False
        self.consist_cnt = {'down': 0, 'mid': 0, 'up': 0}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, i, t, x_s, x_t, x_m, alpha_prod, maps=None):
        if (self.local_blend is not None) and (i > 0):
            use_xm = (self.cur_step + self.start_steps + 1 == self.num_steps)
            x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm, maps=maps)
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out

    def self_attn_forward(self, q, k, v, num_heads):
        if q.shape[0] // num_heads == 3:
            if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                q = torch.cat([q[:num_heads * 2], q[num_heads:num_heads * 2]])
                k = torch.cat([k[:num_heads * 2], k[:num_heads]])
                v = torch.cat([v[:num_heads * 2], v[:num_heads]])
            else:
                q = torch.cat([q[:num_heads], q[:num_heads], q[:num_heads]])
                k = torch.cat([k[:num_heads], k[:num_heads], k[:num_heads]])
                v = torch.cat([v[:num_heads * 2], v[:num_heads]])
            return q, k, v
        else:
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                qu = torch.cat([qu[:num_heads * 2], qu[num_heads:num_heads * 2]])
                qc = torch.cat([qc[:num_heads * 2], qc[num_heads:num_heads * 2]])
                ku = torch.cat([ku[:num_heads * 2], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads * 2], kc[:num_heads]])
                vu = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[:num_heads * 2], vc[:num_heads]])
            else:
                qu = torch.cat([qu[:num_heads], qu[:num_heads], qu[:num_heads]])
                qc = torch.cat([qc[:num_heads], qc[:num_heads], qc[:num_heads]])
                ku = torch.cat([ku[:num_heads], ku[:num_heads], ku[:num_heads]])
                kc = torch.cat([kc[:num_heads], kc[:num_heads], kc[:num_heads]])
                vu = torch.cat([vu[:num_heads * 2], vu[:num_heads]])
                vc = torch.cat([vc[:num_heads * 2], vc[:num_heads]])

            return torch.cat([qu, qc], dim=0), torch.cat([ku, kc], dim=0), torch.cat([vu, vc], dim=0)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross:

            if self.use_consist and attn.shape[1] <= 32 ** 2:
                attn = self.consist_store[place_in_unet + '_cross'][self.consist_cnt[place_in_unet]]
                self.consist_cnt[place_in_unet] += 1

            h = attn.shape[0] // self.batch_size
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])

            attn_base, attn_replace, attn_masa = attn[0], attn[1], attn[2]
            attn_replace_new = self.replace_cross_attention(attn_masa, attn_replace)
            attn_base_store = self.replace_cross_attention(attn_base, attn_replace)
            if (self.cross_replace_steps >= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                attn[1] = attn_base_store

            attn_store = torch.cat([attn_base_store, attn_replace_new])
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
            attn_store = attn_store.reshape(2 * h, *attn_store.shape[2:])

            consist_store = attn

            super(AttentionControlEdit, self).forward(attn_store, is_cross, place_in_unet, consist_store)
        return attn

    def __init__(self, prompts, num_steps: int, start_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts) + 1
        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to("cuda").to(torch.float16)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_masa, att_replace):
        attn_masa_replace = attn_masa[:, :, self.mapper].squeeze()
        attn_replace = attn_masa_replace * self.alphas + \
                       att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, prompt_specifiers, tokenizer, encoder, num_steps: int, start_steps: int,
                 cross_replace_steps: float, self_replace_steps: float, local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, start_steps, cross_replace_steps, self_replace_steps,
                                              local_blend)
        self.mapper, alphas, ms, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers,
                                                                                      tokenizer, encoder, "cuda")
        self.mapper, alphas, ms = self.mapper.to("cuda"), alphas.to("cuda").to(torch.float16), ms.to("cuda").to(
            torch.float16)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to("cuda")
        alpha_e = alpha_e.to("cuda")
        alpha_m = alpha_m.to("cuda")
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(ms, alphas, alpha_e, alpha_m, t_len)


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float], Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float16)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def inference(pipe, img, source_prompt, target_prompt,
              local, mutual,
              positive_prompt, negative_prompt,
              guidance_s, guidance_t,
              num_inference_steps,
              width, height, seed, strength,
              cross_replace_steps, self_replace_steps,
              thresh_e, thresh_m, denoise):
    print(img)

    torch.manual_seed(seed)
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    if denoise is False:
        strength = 1
    num_denoise_num = math.trunc(num_inference_steps * strength)
    num_start = num_inference_steps - num_denoise_num
    # create the CAC controller.
    local_blend = LocalBlend(thresh_e=thresh_e, thresh_m=thresh_m, save_inter=False)
    controller = AttentionRefine([source_prompt, target_prompt], [[local, mutual]], pipe.tokenizer, pipe.text_encoder,
                                 num_inference_steps,
                                 num_start,
                                 cross_replace_steps=cross_replace_steps,
                                 self_replace_steps=self_replace_steps,
                                 local_blend=local_blend
                                 )
    ptp_utils.register_attention_control(pipe, controller)

    results = pipe(prompt=target_prompt,
                   source_prompt=source_prompt,
                   positive_prompt=positive_prompt,
                   negative_prompt=negative_prompt,
                   image=img,
                   num_inference_steps=num_inference_steps,
                   eta=1,
                   strength=strength,
                   guidance_scale=guidance_t,
                   source_guidance_scale=guidance_s,
                   denoise_model=denoise,
                   callback=controller.step_callback
                   )

    return replace_nsfw_images(results)


def replace_nsfw_images(results):
    for i in range(len(results.images)):
        if results.nsfw_content_detected[i]:
            results.images[i] = Image.open("nsfw.png")
    return results.images[0]


if __name__ == "__main__":

    model_id_or_path = "SimianLuo/LCM_Dreamshaper_v7"

    scheduler = LCMScheduler.from_config(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                         subfolder="scheduler")
    pipe = EditPipeline.from_pretrained(model_id_or_path, use_auth_token=os.environ.get("USER_TOKEN"),
                                        scheduler=scheduler, torch_dtype=torch.float16)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    img1 = Image.open("images/man00.png")
    input1 = [pipe,img1,"man","black man","black man","","","",1,2,15,512,512,0,1,0.7,0.7,0.6,0.6,False]
    output1 = inference(*input1).permute(1, 2, 0).to(torch.float32)
    plt.imshow(output1.cpu())
    plt.show()
