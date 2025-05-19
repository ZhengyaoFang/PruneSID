import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLPreTrainedModel, Qwen2VLCausalLMOutputWithPast, Qwen2VLAttention, repeat_kv, apply_multimodal_rotary_pos_emb
from transformers.generation.utils import GenerationMixin
from transformers.utils import (
    replace_return_docstrings,
)
import math
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache

def pca_group(features, min_components=32):
    standard_features = torch.sigmoid(features.to(torch.float32)).permute(1,0)
    U, S, V = torch.pca_lowrank(standard_features, q=min_components)
    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=1)
    return V, belong_components

def nms(similarity_matrix, scores, threshold):
    keep = []
    while scores.sum() > 0:
        max_idx = scores.argmax(axis=0)
        scores[max_idx] = 0
        keep.append(max_idx)
        condition = similarity_matrix[max_idx] > threshold
        scores[condition] = 0
    return keep

class Qwen2VisionTransformerPretrainedModel_prunesid(Qwen2VLPreTrainedModel):
    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        # hidden_states: [grid_thw[0][1] * grid_thw[0][2], 1280]
        hidden_states = self.merger(hidden_states) # [token_num, 3584]

        need_token_num = self.need_token_num if self.need_token_num else 192
        need_token_num = max(round(hidden_states.shape[0] * need_token_num / 576), 16)
        if hidden_states.shape[0] <= 16:
            return hidden_states, None
        projector_lengths, belong_components = pca_group(hidden_states, min_components=max(int(need_token_num / 4), 4))
        projector_scores = projector_lengths.clone()

        projector_mask = belong_components.unsqueeze(1).repeat(1, projector_lengths.shape[1])
        index_map = torch.arange(projector_lengths.shape[1], device=projector_lengths.device).unsqueeze(0).repeat(projector_lengths.shape[0],1)
        weights_mask = torch.where(projector_mask != index_map)
        projector_lengths[weights_mask] = 0
        projector_scores[weights_mask] = 0

        normalized_states = F.normalize(hidden_states, p=2, dim=-1)
        group_similarity = torch.bmm(normalized_states.unsqueeze(0), normalized_states.T.unsqueeze(0))[0] # [batch_size, 576, 576]
        sim_mean = group_similarity.triu(diagonal=0).mean()
        group_similarity = group_similarity.to(torch.float32).cpu().numpy()
        group_idxs = []

        
        ratio = max(need_token_num / ((hidden_states.shape[0]) / 18),1)
        given_scores = torch.arange(projector_lengths.shape[0],0,-1, device=projector_lengths.device, dtype=projector_lengths.dtype)
        for g in range(projector_lengths.shape[1]):
            group_indices = torch.where(belong_components == g)[0].cpu().numpy()
            if group_indices.shape[0] == 0:
                group_idxs.append(np.array([]))
                continue
            g_similarity = group_similarity[group_indices, :][ :, group_indices]
            g_scores = projector_lengths[:,g][group_indices].cpu().numpy()
            keep_indices = nms(g_similarity, g_scores, float(ratio * sim_mean))
            keep_indices = group_indices[keep_indices]
            projector_scores[keep_indices,g] = given_scores[:keep_indices.shape[0]]
            group_idxs.append(keep_indices)    
        
        keep_nms_counts = torch.tensor([group_idxs[i].shape[0] for i in range(len(group_idxs))], device=projector_lengths.device)
        group_counts = (projector_mask == index_map).sum(dim=0)

 
        group_lower_bound = torch.ones(group_counts.shape[0], device=group_counts.device)
        group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]

        group_upper_bound = torch.ones(group_counts.shape[0], device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
        group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0), dim=0)[0]
        while group_upper_bound.sum() < need_token_num:
            group_upper_bound = group_upper_bound + 1
            group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0] 
        
        other_token_nums = max(0, need_token_num - group_lower_bound.sum())
        norm_group_counts = keep_nms_counts / keep_nms_counts.sum()
        cumulative_sum = torch.cumsum(norm_group_counts, dim=0)
        other_token_d = (cumulative_sum * other_token_nums).round().int()
        other_token_d = other_token_d - torch.cat([torch.zeros(1, device=other_token_d.device), other_token_d[:-1]])
        group_token_d = other_token_d + group_lower_bound


        group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
        group_mean_sort_index = torch.argsort(keep_nms_counts, descending=True, dim=0)
        filling_group = 0

        while group_token_d.sum() < other_token_nums + group_lower_bound.sum():
            filling_num = min(group_upper_bound[group_mean_sort_index[filling_group]] - group_token_d[group_mean_sort_index[filling_group]], other_token_nums + group_lower_bound.sum() - group_token_d.sum())
            group_token_d[group_mean_sort_index[filling_group]] += filling_num
            filling_group += 1
        # print(group_token_d.sum(), keep_nms_counts.sum())
        projector_sort_index = torch.argsort(projector_scores, descending=True, dim=0)
        important_indices = []
        belong_components_ = []
        for g in range(len(group_token_d)):
            important_indices.append(projector_sort_index[:,g][:int(group_token_d[g])])
            belong_components_.extend([g]*int(group_token_d[g]))
        important_indices = torch.cat(important_indices, dim=0)[:need_token_num]
        if important_indices.shape[0] > 150:
            important_indices = important_indices.sort()[0]
        return hidden_states, important_indices
    
_CONFIG_FOR_DOC = "Qwen2VLConfig"
    
class Qwen2VLForConditionalGeneration_prunesid(Qwen2VLPreTrainedModel, GenerationMixin):
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        important_ids = None
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds, important_ids = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device)
                if important_ids is not None:
                    important_ids = important_ids.to(inputs_embeds.device)
                image_mask = input_ids == self.config.image_token_id
                if self.training:
                    inputs_embeds = inputs_embeds.clone()
                inputs_embeds[image_mask] = image_embeds
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).to(inputs_embeds.device)
                video_mask = input_ids == self.config.video_token_id
                inputs_embeds[video_mask] = video_embeds
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if important_ids is not None:
            keep_ids = []
            for b in range(input_ids.shape[0]):
                image_start_ids = torch.where(input_ids[b] == self.config.image_token_id)[0][0]
                image_end_ids = torch.where(input_ids[b] == self.config.image_token_id)[0][-1]
                important_ids_ = important_ids + image_start_ids
                keep_ids.append(torch.cat([torch.arange(0, image_start_ids, device=input_ids.device), important_ids_, torch.arange(image_end_ids + 1, input_ids.shape[1], device=input_ids.device)], dim=0))
            assert len(keep_ids) == 1
            position_ids = position_ids[:,:,keep_ids[0]].view(position_ids.shape[0], position_ids.shape[1], -1)
            inputs_embeds = inputs_embeds[:,keep_ids[0]].view(inputs_embeds.shape[0], -1, inputs_embeds.shape[2])
            attention_mask = attention_mask[:,keep_ids[0]].view(attention_mask.shape[0], -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

class Qwen2VLSdpaAttention_prunesid(Qwen2VLAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = max(key_states.shape[-2], position_ids.max()+1)
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value