import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from train_utils.scheduler import (
    get_wsd_lr_lambda,
    get_two_stage_cosine_lr_lambda,
    get_custom_lr_lambda,
)


class CustomTrainer(Trainer):
    exe_mode: str = "train"

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer = None,
    ):
        self.total_steps = num_training_steps

        if optimizer is None:
            optimizer = self.optimizer

        if not self.args.lr_decay_style and self.args.lr_decay_style == "none":
            return super().create_scheduler(num_training_steps, optimizer)
        
        warmup_steps = int(self.args.warmup_ratio * num_training_steps)

        if self.args.lr_decay_style == "wsd":
            lr_lambda = get_wsd_lr_lambda(
                total=num_training_steps,
                wsd_ratio=getattr(self.args, "wsd_ratio", 0.1),
                min_lr_ratio=getattr(self.args, "min_lr_ratio", 0.0),
                warmup=warmup_steps,
            )
        elif self.args.lr_decay_style == "two_stage":
            lr_lambda = get_two_stage_cosine_lr_lambda(
                total=num_training_steps,
                stage_boundary_ratio=getattr(self.args, "stage_ratio", 0.5),
                first_stage_scale=1.0,
                second_stage_scale=0.667,
                min_lr_ratio=getattr(self.args, "min_lr_ratio", 0.006),
                warmup=warmup_steps,
            )
        elif self.args.lr_decay_style == "custom":
            lr_lambda = get_custom_lr_lambda(
                total=num_training_steps,
                stage_boundary_ratio=getattr(self.args, "stage_ratio", 0.5),
                first_stage_scale=1.0,
                warmup=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler style: {self.lr_decay_style}")

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        self._created_lr_scheduler = True
        return self.lr_scheduler

    def get_train_dataloader(self):
        if self.exe_mode != "dry_run_fast":
            return super().get_train_dataloader()

        # Tiny static batch to avoid disk I/O during dry-run.
        pad_id = self.processing_class.pad_token_id or 0
        seq = 8
        example = {
            "input_ids": [pad_id] * seq,
            "attention_mask": [1] * seq,
            "labels": [pad_id] * seq,
        }

        class _DummyDataset(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return example

        dummy_ds = _DummyDataset()

        return DataLoader(
            dummy_ds,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=0,
        )

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):
        if self.exe_mode != "dry_run":
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        return torch.zeros([], device=device, requires_grad=True)


def get_train_args(
        args,
        grad_accum_steps,
        max_steps,
    ):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to=args.report_to,
        max_steps=max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
    )
    training_args.lr_decay_style = args.lr_decay_style
    training_args.wsd_ratio = args.wsd_ratio
    training_args.min_lr_ratio = args.min_lr_ratio
    training_args.stage_ratio = args.stage_ratio

    return training_args


def train(
    model,
    training_args,
    train_dataset,
    data_collator,
    tokenizer,
    output_dir: str,
    callbacks=None,
    resume: bool = True,
    exe_mode: str = "train",
):
    """
    Training orchestration function.
    """
    callbacks = callbacks or []

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    print(trainer.lr_scheduler)
    trainer.exe_mode = exe_mode

    last_ckpt = get_last_checkpoint(output_dir) if resume else None
    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
