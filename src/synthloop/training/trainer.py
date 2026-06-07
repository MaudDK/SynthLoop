from synthloop.model.loader import load_model
from synthloop.config import load_config, authenticate_wandb
from synthloop.model.loras import build_lora_config, apply_lora
from datasets import DatasetDict
from trl import SFTTrainer, SFTConfig
import wandb

def apply_chat_template(dataset, tokenizer) -> DatasetDict:
    def format_row(row):
        row['text'] = tokenizer.apply_chat_template(
            row['messages'],
            tokenize=False,
            add_generation_prompt=False
        )

        return row
    return dataset.map(format_row)

def build_trainer(model, tokenizer, dataset, config):

    sft_config = SFTConfig(
        output_dir = config['model']['output_dir'],
        num_train_epochs = config['training']['epochs'],
        per_device_train_batch_size = config['training']['batch_size'],
        gradient_accumulation_steps = config['training']['gradient_accumulation'],
        learning_rate = config['training']['learning_rate'],
        bf16 = config['training']['bf16'],
        logging_steps = 10,
        save_strategy = "epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model = "eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataset_text_field = "text",
        max_length = config['training']['max_length'],
        report_to = "wandb",
        run_name = config['wandb']['run_name'],
        gradient_checkpointing=True,                            # add
        gradient_checkpointing_kwargs={"use_reentrant": False}, # add
        torch_empty_cache_steps=1,                              
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
    )

    return trainer

def train(config_path: str, dataset):
    # Load config
    config = load_config(config_path)
    authenticate_wandb()
    wandb.init(
        project=config['wandb']['project'], 
        name=config['wandb']['run_name']
    )

    # Load model and tokenizer
    model, tokenizer = load_model(config_path)

    # Build LoRA config and apply to model
    lora_config = build_lora_config(config_path)

    # Apply chat template to dataset
    dataset = apply_chat_template(dataset, tokenizer)

    model = apply_lora(model, lora_config)

    trainer = build_trainer(model, tokenizer, dataset, config)

    try:
        trainer.train()
    finally:
        wandb.finish()

    output_dir = config['model']['output_dir']
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapted model saved to {output_dir}")