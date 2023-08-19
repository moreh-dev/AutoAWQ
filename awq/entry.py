import os
import torch
import argparse
from lm_eval import evaluator
from transformers import AutoTokenizer
from awq.models.auto import AutoAWQForCausalLM
from awq.quantize.auto_clip import apply_clip
from awq.quantize.auto_scale import apply_scale
from awq.utils.lm_eval_adaptor import LMEvalAdaptor


def load_search_result_into_memory(model, search_path):
    awq_results = torch.load(search_path, map_location="cpu")
            
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

def run_search(model_path, dump_path, quant_config):
    """
    Step 1/2: Search the pile for an optimal scaling factor.
    """
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config, run_search=True, run_quant=False)

    # Save search results
    model.save_quantized(dump_path)

    # Save tokenizer
    tokenizer.save_pretrained(dump_path)

def run_quant(model_path, search_path, dump_path, quant_config):
    """
    Step 2/2: Use the search results to quantize model weights
    """
    # Load model and search results
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    load_search_result_into_memory(model.model, search_path)

    # Run actual weight quantization
    model.quantize(quant_config=quant_config, run_search=False, run_quant=True)

    # Save quantized model
    model.save_quantized(dump_path)

def run_perplexity(quant_path, quant_file, quant_config, device):
    """
    Post quantization: Evaluate perplexity on wikitext with EleutherAI Evaluation Harness
    """
    # Load model
    model = AutoAWQForCausalLM.from_quantized(quant_path, quant_file, quant_config)
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

    # Load adapter
    lm_eval_model = LMEvalAdaptor(quant_path, model, tokenizer, device, batch_size=1)

    # Evaluate perplexity of quantized model
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=['wikitext'],
        batch_size=1,
        no_cache=True,
        num_fewshot=0,
    )

    print(evaluator.make_table(results))

if __name__ == '__main__':
    """
    python -m awq.entry --entry_type search --model_path mosaicml/mpt-7b-8k-chat --search_path mpt-7b-8k-chat-awq
    python -m awq.entry --entry_type quant --model_path mosaicml/mpt-7b-8k-chat --search_path mpt-7b-8k-chat-awq/awq_model_search_result.pt --quant_path mpt-7b-8k-chat-awq
    python -m awq.entry --entry_type perplexity --quant_path mpt-7b-8k-chat-awq --quant_file awq_model_w4_g128.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry_type', type=str, help='The type of task to run (search|quant|perplexity)')
    parser.add_argument('--model_path', type=str, help='Path to hf model')
    parser.add_argument('--search_path', type=str, help='Path to save/load AWQ search results')
    parser.add_argument('--quant_path', type=str, help='Path to AWQ model directory')
    parser.add_argument('--quant_file', type=str, help='Path to quantized AWQ model file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to load model to')
    parser.add_argument('--w_bit', type=int, default=4)
    parser.add_argument('--q_group_size', type=int, default=128)
    args = parser.parse_args()

    quant_config = { "zero_point": True, "q_group_size": args.q_group_size, "w_bit": args.w_bit }
    
    if args.entry_type == 'search':
        run_search(args.model_path, args.search_path, quant_config)
    elif args.entry_type == 'quant':
        run_quant(args.model_path, args.search_path, args.quant_path, quant_config)
    elif args.entry_type == 'perplexity':
        run_perplexity(args.quant_path, args.quant_file, args.w_bit, quant_config, args.device)
    else:
        raise Exception('--entry_type must be one of (search|quant|perplexity)')