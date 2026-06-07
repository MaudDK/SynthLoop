from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from synthloop.config import load_config

def build_lora_config(config_path: str) -> LoraConfig:
    config = load_config(config_path)

    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def apply_lora(model, lora_config: LoraConfig):
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model