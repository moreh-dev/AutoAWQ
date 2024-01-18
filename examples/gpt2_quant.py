from awq import AutoAWQForCausalLM
from awq.utils.utils import replace_conv1d_with_linear

from transformers import AutoTokenizer

model_path = "gpt2"
quant_path = "gpt2-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
replace_conv1d_with_linear(model)
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, use_fast=False
)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
