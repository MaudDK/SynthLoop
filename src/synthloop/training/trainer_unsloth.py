from unsloth import FastLanguageModel
from synthloop.config import load_config, authenticate_wandb
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


def load_model_unsloth(config: dict):
    """Load model and tokenizer using Unsloth for faster training."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['base_model'],
        max_seq_length=config['training']['max_length'],
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer


def apply_lora_unsloth(model, config: dict):
    """Apply LoRA adapters using Unsloth's optimised implementation."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model


def build_trainer(model, tokenizer, dataset, config):
    sft_config = SFTConfig(
        output_dir=config['model']['output_dir'],
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation'],
        learning_rate=config['training']['learning_rate'],
        bf16=config['training']['bf16'],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataset_text_field="text",
        max_length=config['training']['max_length'],
        report_to="wandb",
        run_name=config['wandb']['run_name'],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
    )

    return trainer


def train(config_path: str, dataset: DatasetDict, iteration: int = 0):
    config = load_config(config_path)
    authenticate_wandb()

    model, tokenizer = load_model_unsloth(config)
    model = apply_lora_unsloth(model, config)
    model.print_trainable_parameters()
    dataset = apply_chat_template(dataset, tokenizer)

    wandb.init(
        project=config['wandb']['project'],
        name=f"{config['wandb']['run_name']}-iter-{iteration}"
    )

    trainer = build_trainer(model, tokenizer, dataset, config)

    try:
        trainer.train()
    finally:
        wandb.finish()

    output_dir = f"{config['model']['output_dir']}/iter_{iteration}"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")