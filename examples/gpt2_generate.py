from awq import AutoAWQForCausalLM

from transformers import AutoTokenizer, TextStreamer

quant_path = "gpt2-awq"

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(
    quant_path, trust_remote_code=True, use_fast=False
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
model.generate(encoded_input, streamer=streamer, max_new_tokens=20)
