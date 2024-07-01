from dataclasses import dataclass
from typing import Any, List, Dict, Union

from datasets import load_dataset
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainer
import torchaudio
import evaluate
import torch
import os


def dataset_loader(batch):
    signal, fs = torchaudio.load(batch["audio"])
    batch["Audio"] = signal
    with open(batch["text"], "r") as f:
        transcribe = f.read()
    batch["Text"] = transcribe
    return batch


def prepare_dataset(batch):
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Indonesian", task="transcribe")
    audio = batch["Audio"]
    batch["input_features"] = feature_extractor(audio, sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["Text"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }


def compute_metrics(pred):
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Indonesian", task="transcribe")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_token=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_token=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == "__main__":
    DIR = "json_directory"
    data_files = {"train": os.path.join(DIR, "train.jsonl"), "test": os.path.join(DIR, "test.jsonl")}
    dataset = load_dataset("json", data_files=data_files)

    dataset = dataset.map(
        dataset_loader,
        remove_columns=dataset.column_names["train"],
        num_proc=2
    )

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=2
    )

    processor = WhisperProcessor.from_pretrained(
        pretrained_model_name_or_path="openai/whisper-small",
        language="Indonesian",
        task="transcribe"
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.language = "Indonesian"
    model.config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-ina",
        per_device_train_batch_size=5,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=10,
        max_steps=100,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=20,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Training Started.")
    best_trial = trainer.hyperparameter_search(
        direction=["minimize", "maximize"],
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=20,
    )
    trainer.train()
    trainer.save_model()
    print("Training Done.")
