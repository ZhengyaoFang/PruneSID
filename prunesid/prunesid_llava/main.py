from .clip_encoder import CLIPVisionTower_PruneSID
from .llava_arch import prepare_inputs_labels_for_multimodal_prunesid, encode_images_prunesid, encode_images_prunesid_multi, restore_image_features_sorted

def prunesid_llava(model, need_token_num=64):
    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_PruneSID.forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_prunesid
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_prunesid_multi = encode_images_prunesid_multi
        LlavaMetaForCausalLM.encode_images_prunesid = encode_images_prunesid
    model.model.vision_tower.need_token_num = need_token_num
    return model