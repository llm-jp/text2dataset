import datasets
from datasets import load_dataset
import os
from datasets import Dataset
import wandb
import time

from text2dataset.translator import Translator, DeeplTranslator, OpenAIAPITranslator
import logging
import json
from text2dataset.writer import write_shard
from text2dataset.utils import State
from text2dataset.reader import create_dataset
import yaml
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Args:
    model_id: str
    batch_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_id: int
    input_path: str
    source_column: str
    target_column: str
    push_to_hub: bool
    push_to_hub_path: str
    output_dir: str
    output_format: str
    number_sample_per_shard: int
    resume_from_checkpoint: bool
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    prompt_template_path: str
    temperature: float
    top_p: float
    max_tokens: int
    target_lang: str
    keep_columns: str | None
    split: str


def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for model inference")
    parser.add_argument(
        "--model_id",
        type=str,
        default="llm-jp/llm-jp-3-3.7b-instruct",
        help="Model name. e.g., llm-jp/llm-jp-3-3.7b-instruct. Specify 'gpt-4o-mini-2024-07-18' for OpenAI API or 'deepl' for DeepL API.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for vLLM inference."
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/english_quotes.json",
        help="Local file path or Hugging Face dataset name.",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="txt",
        help="Column name in the dataset to be prompted.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="txt_ja",
        help="Column name in the dataset to store generated text.",
    )
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--push_to_hub_path", type=str, default="speed/english_quotes")
    parser.add_argument("--output_dir", type=str, default="data/english_quotes_ja")
    parser.add_argument("--output_format", type=str, default="json")
    parser.add_argument("--number_sample_per_shard", type=int, default=1000)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=bool,
        default=False,
        help="Resume from the last checkpoint.",
    )
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="text2dataset")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default="config/prompt.yaml",
        help="Path to the prompt template.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument(
        "--target_lang",
        type=str,
        default="ja",
        help="Target language for translation; used for DeepL API.",
    )
    parser.add_argument(
        "--keep_columns",
        type=str,
        default=None,
        help="Columns to keep in the output dataset, separated by comma. If None, all columns are kept. e.g., 'txt'. target_column is always kept.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to use. e.g., 'train', 'validation', 'test'.",
    )

    args = parser.parse_args()
    return Args(**vars(args))


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main():
    args = parse_args()

    # Text in source_column of the Dataset will be translated into Japanese.
    state = State(0, 0, 0)
    if args.resume_from_checkpoint:
        state_path = os.path.join(args.output_dir, "state.jsonl")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = State(**json.load(f), total_processed_examples=0)
            logger.info(
                f"Resuming from {state.current_shard_id} shard and {state.last_saved_example_num} example"
            )
        else:
            logger.info("No state file found. Starting from scratch")
        # reset state.jsonl

    logger.info("Start translation")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.makedirs(args.output_dir, exist_ok=True)
    state_path = os.path.join(args.output_dir, "state.jsonl")
    ds = create_dataset(args.input_path, state, args.split)
    # keep only the specified columns
    if args.keep_columns is not None:
        ds = ds.select_columns(args.keep_columns.split(","))
    # batch dataloader
    data_loader = ds.batch(batch_size=args.batch_size)

    if args.use_wandb:
        config_parameters = dict(locals())
        config_parameters.pop("use_wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config_parameters,
        )

    with open(args.prompt_template_path) as f:
        data = yaml.safe_load(f)
        template = data["prompt"]

    if args.model_id == "deepl":
        translator = DeeplTranslator(args.target_lang)
    elif args.model_id in ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-07-18"]:
        translator = OpenAIAPITranslator(
            args.model_id, template, args.temperature, args.top_p, args.max_tokens
        )
    else:
        translator = Translator(
            args.model_id,
            args.tensor_parallel_size,
            args.pipeline_parallel_size,
            template,
            args.temperature,
            args.top_p,
            args.max_tokens,
        )

    dataset_buffer = Dataset.from_dict({})

    for examples in data_loader:
        start_time = time.time()
        text_list = examples[args.source_column]
        translated = translator.translate(text_list)
        # store to buffer
        dataset_buffer = datasets.concatenate_datasets(
            [
                dataset_buffer,
                datasets.Dataset.from_dict(
                    {**examples, args.target_column: translated}
                ),
            ]
        )
        state.total_processed_examples += len(text_list)
        examples_per_sec = len(text_list) / (time.time() - start_time)

        # write shards to output_dir if the buffer is full
        # e.g number_sample_per_shard = 100, len(dataset_buffer) = 1024
        # 1024 // 100 = 10 shards will be written to output_dir
        if len(dataset_buffer) >= args.number_sample_per_shard:
            for i in range(len(dataset_buffer) // args.number_sample_per_shard):
                shard_dict = dataset_buffer[
                    i * args.number_sample_per_shard : (i + 1)
                    * args.number_sample_per_shard
                ]
                shard_ds = Dataset.from_dict(shard_dict)

                state = write_shard(
                    shard_ds, args.output_dir, args.output_format, state
                )
                state.current_shard_id += 1
                state.save_state(state_path)

            dataset_buffer = Dataset.from_dict(
                dataset_buffer[
                    len(dataset_buffer)
                    // args.number_sample_per_shard
                    * args.number_sample_per_shard :
                ]
            )

        if wandb.run is not None:
            wandb.log(
                {
                    "count": state.total_processed_examples,
                    "examples_per_sec": examples_per_sec,
                }
            )
        # write shards if the queue is full

    # write the remaining examples
    if len(dataset_buffer) > 0:
        state = write_shard(dataset_buffer, args.output_dir, args.output_format, state)
        state.save_state(state_path)

    if args.push_to_hub:
        if args.output_format == "jsonl" or args.output_format == "json":
            # jsonl without state.jsonl
            files = os.listdir(args.output_dir)
            if "state.jsonl" in files:
                files.remove("state.jsonl")
            # Sort files by shard id to keep the order.
            files.sort(key=lambda x: int(x.split(".")[0]))
            translated_ds = load_dataset(
                "json", data_files=[os.path.join(args.output_dir, f) for f in files]
            )
        elif args.output_format == "parquet":
            translated_ds = load_dataset(
                "parquet", data_files=os.path.join(args.output_dir, "*.parquet")
            )
        translated_ds.push_to_hub(args.push_to_hub_path, private=True)


if __name__ == "__main__":
    main()
