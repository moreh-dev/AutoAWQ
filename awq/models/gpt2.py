from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from .base import BaseAWQForCausalLM


class GPT2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "GPT2Block"
    max_new_tokens_key = "n_positions"

    @staticmethod
    def get_model_layers(model: GPT2LMHeadModel):
        return model.transformer.h

    @staticmethod
    def get_act_for_scaling(module: GPT2Block):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.c_fc.out_features
        )

    @staticmethod
    def move_embed(model: GPT2LMHeadModel, device):
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)

    @staticmethod
    def get_layers_for_scaling(module:GPT2Block, input_feat, module_kwargs):
        layers = []

        # linear 1
        layers.append(dict(
            prev_op=module.ln_2,
            layers=[module.mlp.c_fc],
            inp=input_feat['mlp.c_fc'],
            module2inspect=module.mlp
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.c_proj],
            inp=input_feat['mlp.c_proj']
        ))

        return layers
