from .modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel_prunesid, Qwen2VLForConditionalGeneration_prunesid, Qwen2VLSdpaAttention_prunesid

def prunesid_qwen2(model, need_token_num=64):
    model.visual.need_token_num = need_token_num
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    Qwen2VisionTransformerPretrainedModel.forward = Qwen2VisionTransformerPretrainedModel_prunesid.forward
    
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    Qwen2VLForConditionalGeneration.forward = Qwen2VLForConditionalGeneration_prunesid.forward

    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLSdpaAttention
    Qwen2VLSdpaAttention.forward = Qwen2VLSdpaAttention_prunesid.forward
    return model