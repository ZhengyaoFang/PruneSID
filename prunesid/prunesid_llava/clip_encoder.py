import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
import math

def batch_similarity_nms(similarity_matrix, scores, threshold):
    """
    similarity_matrix: shape [batch_size, group, N, N]
    scores: shape [batch_size, N, group]
    threshold: [batch_size]
    """
    new_scores = scores.clone()
    return_scores = new_scores.clone()
    given_score = 1000
    keep = []

    batch_size, group, N, _ = similarity_matrix.shape
    threshold = threshold.unsqueeze(1).unsqueeze(2).expand(batch_size, group, N)
    batch_idx= torch.arange(batch_size, device=new_scores.device).view(-1, 1, 1).expand(-1, group, N)
    group_idx = torch.arange(group, device=new_scores.device).view(1, -1, 1).expand(batch_size, -1, N)   # [5, 16, 576]
    col_idx = torch.arange(N, device=new_scores.device).view(1, 1, -1).expand(batch_size, group, -1)     # [5, 16, 576]
    while new_scores.sum() > 0:
        max_values, max_idx = new_scores.max(dim=1) # [batch_size, group], [batch_size, group]

        row_idx = max_idx.unsqueeze(-1).expand(-1, -1, N)
        sim_row = similarity_matrix[batch_idx, group_idx, row_idx, col_idx] # [batch_size, group, N]
        max_idx[max_values == 0] = -1
        given_score_indices_0, given_score_indices_2 = torch.where(max_idx!=-1)
        given_score_indices_1 = max_idx[given_score_indices_0, given_score_indices_2]
        return_scores[given_score_indices_0, given_score_indices_1, given_score_indices_2] = given_score
        new_scores[given_score_indices_0, given_score_indices_1, given_score_indices_2] = 0
        given_score -= 1
        keep.append(max_idx.unsqueeze(-1))

        condition = sim_row > threshold # [batch_size, group, N]
        new_scores[condition.transpose(1,2)] = 0
        if new_scores.sum() == 0:
            break
    keep = torch.cat(keep, dim=-1) # [5, 16, ???]
    return keep, return_scores
    

def batch_pca(features, min_components=32):
    standard_features = torch.sigmoid(features.to(torch.float32)).transpose(2,1)[:,:,1:]
    U, S, V = torch.pca_lowrank(standard_features, q=min_components)
    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=-1)
    return V, belong_components


class CLIPVisionTower_PruneSID(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        # images: torch.Tensor [batch_size, 3, 336, 336]
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2] # [1, 577, 1024]

            need_token_num = self.need_token_num if self.need_token_num else 192
            projector_lengths, belong_components = batch_pca(hidden_states, min_components = int(need_token_num / 4)) # [batch_size, 576, group_num], [batch_size, 576]
            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx + 1 :]
            cls_attention_sum = cls_attention.sum(dim=1) # [batch_size, 576]
            projector_scores = cls_attention_sum.unsqueeze(-1).repeat(1,1, projector_lengths.shape[-1]).to(projector_lengths.dtype) # [batch_size, 576, group_num]
            projector_mask = belong_components.unsqueeze(-1).repeat(1,1, projector_lengths.shape[-1])
            index_map = torch.arange(projector_lengths.shape[-1], device=projector_lengths.device).unsqueeze(0).unsqueeze(0).repeat(projector_lengths.shape[0],projector_lengths.shape[1],1)
            weights_mask = torch.where(projector_mask != index_map)
            projector_lengths[weights_mask] = 0
            projector_scores[weights_mask] = 0
           
            normalized_states = F.normalize(hidden_states[:,1:,:], p=2,dim=-1)
            similarity = torch.bmm(normalized_states, normalized_states.transpose(2,1)).to(torch.float32) # [batch_size， 576， 576]
            triu_mask = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
            sim_mean = (similarity * (triu_mask)).sum(-1).sum(-1) / triu_mask.sum(-1).sum(-1)
            ratio = need_token_num / 32

            
            group_similarity = similarity.clone().unsqueeze(1).repeat(1, projector_lengths.shape[-1], 1, 1) # [batch_size, group_num, 576, 576]
            group_similarity_masks = torch.zeros_like(group_similarity) # [batch_size, group_num, 576, 576]
            group_index = torch.arange(group_similarity.shape[1], device=group_similarity.device).unsqueeze(0).unsqueeze(-1).repeat(group_similarity.shape[0],1,group_similarity.shape[-2])
            group_belong_components = belong_components.unsqueeze(1).repeat(1,group_similarity.shape[1],1)
            group_similarity_masks[group_index==group_belong_components] = 1
            group_similarity_masks[torch.where(group_similarity_masks.transpose(3,2) == 1)] = 1
            group_similarity[group_similarity_masks!=1] = 0
            group_ids, projector_scores = batch_similarity_nms(group_similarity, projector_scores, ratio * sim_mean)
            group_ids_mask = (group_ids != -1) # [batch_size, group, ???]
            keep_nms_counts = group_ids_mask.sum(dim=-1) # [batch_size, group]
            group_counts = (projector_mask == index_map).sum(dim=1) # [batch_size, group]
            group_lower_bound = torch.ones_like(group_counts, device=group_counts.device)
            group_lower_bound = torch.min(torch.cat([group_lower_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
            group_upper_bound = torch.ones_like(group_counts, device=group_counts.device) * 5 * math.ceil(need_token_num / 64)
            group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]
            group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), keep_nms_counts.unsqueeze(0)], dim=0), dim=0)[0]
            while torch.any(group_upper_bound.sum(-1) < need_token_num):
                group_upper_bound[torch.where(group_upper_bound.sum(-1) < need_token_num)] += 1
                group_upper_bound = torch.min(torch.cat([group_upper_bound.unsqueeze(0), group_counts.unsqueeze(0)], dim=0), dim=0)[0]

            other_token_nums = need_token_num - group_lower_bound.sum(-1) - 1
            other_token_nums[other_token_nums < 0] = 0
            norm_group_counts = keep_nms_counts / keep_nms_counts.sum(-1, keepdim=True)
            cumulative_sum = torch.cumsum(norm_group_counts, dim=-1)
            other_token_d = (cumulative_sum * other_token_nums.unsqueeze(-1).expand(-1, group_counts.shape[1])).round().int()
            other_token_d = other_token_d - torch.cat([torch.zeros((other_token_d.shape[0], 1), device=other_token_d.device), other_token_d[:, :-1]], dim=-1)
            group_token_d = other_token_d + group_lower_bound
            group_token_d = torch.min(torch.cat([group_token_d.unsqueeze(0), group_upper_bound.unsqueeze(0)], dim=0), dim=0)[0]
            group_sort_index = torch.argsort(keep_nms_counts, dim=-1, descending=True) # [batch, group]
            filling_group = torch.zeros(group_counts.shape[0], device=group_counts.device).int()
            while torch.any(group_token_d.sum(-1) < other_token_nums+group_lower_bound.sum(-1)):
                need_filling_batch = torch.where(group_token_d.sum(-1) < other_token_nums+group_lower_bound.sum(-1))[0]
                filling_num = torch.min(torch.stack(
                    [group_upper_bound[need_filling_batch,group_sort_index[need_filling_batch, filling_group[need_filling_batch]]] - group_token_d[need_filling_batch,group_sort_index[need_filling_batch, filling_group[need_filling_batch]]],
                    other_token_nums[need_filling_batch]+group_lower_bound[need_filling_batch].sum(-1)-group_token_d[need_filling_batch].sum(-1)]
                ),dim=0)[0]
                
                group_token_d[need_filling_batch, group_sort_index[need_filling_batch,filling_group[need_filling_batch]]] += filling_num
                filling_group[need_filling_batch] += 1
            projector_sort_index = torch.argsort(projector_scores, dim=1, descending=True) #[batch_size, 576, group]
            projector_sort_index = projector_sort_index.transpose(1,2).reshape(-1, group_similarity.shape[-1]) #[batch_size*group, 576]
            group_token_d = group_token_d.reshape(-1)
            important_indices = []
            for i in range(len(group_token_d)):
                important_indices.append(projector_sort_index[i][:int(group_token_d[i])])

            important_indices = [important_indices[i:i+group_similarity.shape[1]] for i in range(0, len(important_indices), group_similarity.shape[1])]
            for i in range(len(important_indices)):
                important_indices[i] = torch.cat([torch.tensor([0], device=group_similarity.device), torch.cat(important_indices[i])+1])
            batch_indices = torch.stack(important_indices)
            batch_indices_expanded = batch_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)) 
            batch_hidden_states = torch.gather(hidden_states, dim=1, index=batch_indices_expanded)

        return batch_hidden_states, batch_indices # torch.Tensor: [batch_size, token_num, hidden_dim] torch.Tensor: [batch_size, dominant_token_num]