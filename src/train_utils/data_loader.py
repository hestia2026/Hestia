from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from utils.logging import log_progress


def build_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_train_dataset(
    data_path: str,
    seed: int = 42,
):
    dataset = load_from_disk(data_path)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def build_data_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return data_collator


def data_util(
    accelerator,
    logger,
    load_dir,
    data_path,
    random_seed,
    tokenizer_dir=None,
):
    if data_path is None:
        raise ValueError("data_path must be provided for training")

    log_progress(accelerator, logger, "===== Tokenizer & Data Preparing ⏳ =====")
    tokenizer_path = tokenizer_dir or load_dir
    tokenizer = build_tokenizer(tokenizer_path)
    data_collator = build_data_collator(tokenizer)
    train_dataset = build_train_dataset(data_path, random_seed)
    log_progress(accelerator, logger, "===== Data Ready ✅ =====")
    return tokenizer, data_collator, train_dataset
