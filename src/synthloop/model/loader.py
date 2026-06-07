from synthloop.config import load_config, authenticate_hf
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def build_bnb_config(quant_config) -> BitsAndBytesConfig:
    if quant_config['bits'] not in [4, 8]:
        raise ValueError(f"Unsupported quantization bits: {quant_config['bits']}")
    
    if quant_config['bits'] == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE_MAP[quant_config['compute_dtype']],
            bnb_4bit_quant_type=quant_config['quant_type'],
            bnb_4bit_use_double_quant=quant_config['double_quant'],
        )
    else:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
def load_model(config_path: str, inference: bool = False):
    authenticate_hf()
    config = load_config(config_path)
    base_model = config['model']['base_model']
    quantization_config = config['quantization']

    bnb_config = build_bnb_config(quantization_config) if quantization_config['quantize'] else None

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=DTYPE_MAP[quantization_config['compute_dtype']],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        padding_side="left" if inference else "right"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer