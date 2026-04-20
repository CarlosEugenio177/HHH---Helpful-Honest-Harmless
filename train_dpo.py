import argparse
import json
import locale
import os
from pathlib import Path as _Path
from pathlib import Path
from typing import Any

# Force UTF-8 lookups on Windows so TRL templates don't break under cp1252.
if os.name == "nt":
    locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"

    _original_read_text = _Path.read_text

    def _utf8_read_text(self, encoding=None, errors=None, newline=None):
        return _original_read_text(
            self,
            encoding=encoding or "utf-8",
            errors=errors,
            newline=newline,
        )

    _Path.read_text = _utf8_read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino DPO para alinhamento HHH no Laboratorio 08.")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM-135M-Instruct")
    parser.add_argument("--dataset-path", default="dataset/hhh_preferences.jsonl")
    parser.add_argument("--train-dataset-path", default=None)
    parser.add_argument("--eval-dataset-path", default=None)
    parser.add_argument("--sft-adapter", default="adapters/smoke-test")
    parser.add_argument("--output-dir", default="adapters/lab8-dpo")
    parser.add_argument("--cache-dir", default=".hf-cache")
    parser.add_argument("--validation-report-path", default="reports/lab8_validation_summary.json")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-index", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_jsonl_file(path: Path) -> None:
    required_keys = {"prompt", "chosen", "rejected"}
    if not path.exists():
        raise SystemExit(f"Arquivo de dataset nao encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            if set(record.keys()) != required_keys:
                raise SystemExit(
                    f"Dataset invalido em {path} linha {line_number}. "
                    f"As colunas devem ser estritamente prompt, chosen e rejected."
                )


def resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train_dataset_path and args.eval_dataset_path:
        train_path = Path(args.train_dataset_path)
        eval_path = Path(args.eval_dataset_path)
    else:
        dataset_path = Path(args.dataset_path)
        dataset_dir = dataset_path.parent
        train_candidate = dataset_dir / "hhh_preferences_train.jsonl"
        eval_candidate = dataset_dir / "hhh_preferences_test.jsonl"
        if train_candidate.exists() and eval_candidate.exists():
            train_path = train_candidate
            eval_path = eval_candidate
        else:
            train_path = dataset_path
            eval_path = dataset_path

    validate_jsonl_file(train_path)
    validate_jsonl_file(eval_path)
    return train_path, eval_path


def build_quantization_config(torch_module, bits_and_bytes_cls):
    return bits_and_bytes_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_module.float16,
        bnb_4bit_use_double_quant=True,
    )


def configure_cache_dirs(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir.resolve())
    os.environ["HUGGINGFACE_HUB_CACHE"] = str((cache_dir / "hub").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((cache_dir / "transformers").resolve())


def load_tokenizer(auto_tokenizer_cls, model_name: str, cache_dir: Path, trust_remote_code: bool):
    tokenizer = auto_tokenizer_cls.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        cache_dir=str(cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(auto_model_cls, quantization_config, model_name: str, cache_dir: Path, trust_remote_code: bool):
    model = auto_model_cls.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        cache_dir=str(cache_dir),
    )
    model.config.use_cache = False
    return model


def maybe_load_adapter(peft_model_cls, model, adapter_path: Path, is_trainable: bool):
    if adapter_path.exists():
        return peft_model_cls.from_pretrained(model, str(adapter_path), is_trainable=is_trainable)
    return model


def build_default_lora_config(lora_config_cls, task_type_cls):
    return lora_config_cls(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=task_type_cls.CAUSAL_LM,
        target_modules="all-linear",
    )


def format_prompt(prompt: str) -> str:
    return f"### Instrução:\n{prompt.strip()}\n\n### Resposta:\n"


def average_logprob(torch_module, model, tokenizer, prompt: str, completion: str) -> float:
    prefix = format_prompt(prompt)
    full_text = prefix + completion.strip()

    prefix_tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

    input_ids = full_tokens["input_ids"].to(model.device)
    attention_mask = full_tokens["attention_mask"].to(model.device)
    prefix_length = prefix_tokens["input_ids"].shape[1]

    with torch_module.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        log_probs = torch_module.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    completion_start = max(prefix_length - 1, 0)
    completion_token_log_probs = token_log_probs[:, completion_start:]
    if completion_token_log_probs.numel() == 0:
        return float("-inf")
    return completion_token_log_probs.mean().item()


def generate_response(torch_module, model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    encoded = tokenizer(format_prompt(prompt), return_tensors="pt").to(model.device)
    with torch_module.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def save_validation_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Dependencias ausentes. Ative o ambiente virtual e rode `pip install -r requirements.txt`."
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA nao esta disponivel neste ambiente Python. O treino DPO com bitsandbytes exige GPU CUDA visivel."
        )

    train_path, eval_path = resolve_dataset_paths(args)
    cache_dir = Path(args.cache_dir)
    configure_cache_dirs(cache_dir)

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "test": str(eval_path)},
    )

    quantization_config = build_quantization_config(torch, BitsAndBytesConfig)
    tokenizer = load_tokenizer(AutoTokenizer, args.model_name, cache_dir, args.trust_remote_code)

    actor_base = load_base_model(
        AutoModelForCausalLM,
        quantization_config,
        args.model_name,
        cache_dir,
        args.trust_remote_code,
    )
    actor_base = prepare_model_for_kbit_training(actor_base)

    adapter_path = Path(args.sft_adapter)
    use_existing_sft_adapter = adapter_path.exists()
    actor_model = maybe_load_adapter(PeftModel, actor_base, adapter_path, is_trainable=True)
    actor_model.config.use_cache = False

    ref_base = load_base_model(
        AutoModelForCausalLM,
        quantization_config,
        args.model_name,
        cache_dir,
        args.trust_remote_code,
    )
    ref_model = maybe_load_adapter(PeftModel, ref_base, adapter_path, is_trainable=False)
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False

    peft_config = None
    actor_source = f"adaptador SFT em {adapter_path}"
    ref_source = f"adaptador SFT congelado em {adapter_path}"
    if not use_existing_sft_adapter:
        peft_config = build_default_lora_config(LoraConfig, TaskType)
        actor_source = "LoRA nova criada no proprio treino DPO"
        ref_source = "modelo base puro congelado"

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.save_steps,
        report_to="none",
        seed=args.seed,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
    )

    trainer = DPOTrainer(
        model=actor_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_records = list(dataset["test"])
    validation_index = max(0, min(args.validation_index, len(eval_records) - 1))
    validation_example = eval_records[validation_index]

    chosen_score = average_logprob(
        torch,
        trainer.model,
        tokenizer,
        validation_example["prompt"],
        validation_example["chosen"],
    )
    rejected_score = average_logprob(
        torch,
        trainer.model,
        tokenizer,
        validation_example["prompt"],
        validation_example["rejected"],
    )
    generated_text = generate_response(
        torch,
        trainer.model,
        tokenizer,
        validation_example["prompt"],
    )

    validation_report = {
        "model_name": args.model_name,
        "train_dataset_path": str(train_path),
        "eval_dataset_path": str(eval_path),
        "actor_source": actor_source,
        "reference_source": ref_source,
        "beta": args.beta,
        "prompt": validation_example["prompt"],
        "chosen": validation_example["chosen"],
        "rejected": validation_example["rejected"],
        "chosen_avg_logprob": chosen_score,
        "rejected_avg_logprob": rejected_score,
        "difference": chosen_score - rejected_score,
        "safe_preference_reinforced": chosen_score > rejected_score,
        "generated_text": generated_text,
    }
    save_validation_report(Path(args.validation_report_path), validation_report)

    print(f"Adaptador DPO salvo em: {Path(args.output_dir).resolve()}")
    print("Validacao DPO")
    print(f"Ator: {actor_source}")
    print(f"Referencia: {ref_source}")
    print(f"Prompt: {validation_example['prompt']}")
    print(f"Chosen avg logprob: {chosen_score:.4f}")
    print(f"Rejected avg logprob: {rejected_score:.4f}")
    print(f"Diferenca (chosen - rejected): {chosen_score - rejected_score:.4f}")
    print(f"Preferencia segura reforcada: {chosen_score > rejected_score}")
    print("Geracao do modelo alinhado:")
    print(generated_text)
    print(f"Relatorio salvo em: {Path(args.validation_report_path).resolve()}")


if __name__ == "__main__":
    main()
