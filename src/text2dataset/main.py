import datasets
from datasets import load_dataset
import click
import os
from datasets import Dataset
import wandb
import time

from text2dataset.translator import Translator
import logging
import json
from text2dataset.writer import write_shard
from text2dataset.utils import State
from text2dataset.reader import create_dataset
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


@click.command()
@click.option("--model_id", type=str, default="llm-jp/llm-jp-3-3.7b-instruct")
@click.option(
    "--batch_size", type=int, default=1024, help="Batch size for vLLM inference."
)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--pipeline_parallel_size", type=int, default=1)
@click.option("--gpu_id", type=int, default=0)
@click.option(
    "--input_path",
    type=str,
    default="data/english_quotes.json",
    help="Local file path or Hugging Face dataset name.",
)
@click.option(
    "--source_column",
    type=str,
    default="txt",
    help="Existing column name in the dataset to be prompted.",
)
@click.option(
    "--target_column",
    type=str,
    default="txt_ja",
    help="New column name in the dataset to store the generated text.",
)
@click.option("--push_to_hub", type=bool, default=False)
@click.option("--push_to_hub_path", type=str, default="speed/english_quotes")
@click.option("--output_dir", type=str, default="data/english_quotes_ja")
@click.option("--output_format", type=str, default="json")
@click.option("--number_sample_per_shard", type=int, default=1000)
@click.option(
    "--resume_from_checkpoint",
    type=bool,
    default=False,
    help="Resume from the last checkpoint.",
)
@click.option("--use_wandb", type=bool, default=False)
@click.option("--wandb_project", type=str, default="text2dataset")
@click.option("--wandb_run_name", type=str, default="")
@click.option(
    "--prompt_template_path",
    type=str,
    default="config/prompt.yaml",
    help="Path to the prompt template.",
)
@click.option("--temperature", type=float, default=0.8)
@click.option("--top_p", type=float, default=0.95)
@click.option("--max_tokens", type=int, default=200)
def main(
    model_id: str,
    batch_size: int,
    output_dir: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    gpu_id: int,
    source_column: str,
    target_column: str,
    input_path: str,
    push_to_hub: bool,
    push_to_hub_path: str,
    output_format: str,
    number_sample_per_shard: int,
    resume_from_checkpoint: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str,
    prompt_template_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    # Text in source_column of the Dataset will be translated into Japanese.
    state = State(0, 0, 0)
    if resume_from_checkpoint:
        state_path = os.path.join(output_dir, "state.jsonl")
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "state.jsonl")
    ds = create_dataset(input_path, state)
    # batch dataloader
    data_loader = ds["train"].batch(batch_size=batch_size)

    if use_wandb:
        config_parameters = dict(locals())
        config_parameters.pop("use_wandb")
        wandb.init(project=wandb_project, name=wandb_run_name, config=config_parameters)

    with open(prompt_template_path) as f:
        data = yaml.safe_load(f)
        template = data["prompt"]
    translator = Translator(
        model_id,
        tensor_parallel_size,
        pipeline_parallel_size,
        template,
        temperature,
        top_p,
        max_tokens,
    )

    dataset_buffer = Dataset.from_dict({})

    for examples in data_loader:
        start_time = time.time()
        text_list = examples[source_column]
        translated = translator.translate(text_list)
        # store to buffer
        dataset_buffer = datasets.concatenate_datasets(
            [
                dataset_buffer,
                datasets.Dataset.from_dict({**examples, target_column: translated}),
            ]
        )
        state.total_processed_examples += len(text_list)
        examples_per_sec = len(text_list) / (time.time() - start_time)

        # write shards to output_dir if the buffer is full
        # e.g number_sample_per_shard = 100, len(dataset_buffer) = 1024
        # 1024 // 100 = 10 shards will be written to output_dir
        if len(dataset_buffer) >= number_sample_per_shard:
            for i in range(len(dataset_buffer) // number_sample_per_shard):
                shard_dict = dataset_buffer[
                    i * number_sample_per_shard : (i + 1) * number_sample_per_shard
                ]
                shard_ds = Dataset.from_dict(shard_dict)

                state = write_shard(shard_ds, output_dir, output_format, state)
                state.current_shard_id += 1
                state.save_state(state_path)

            dataset_buffer = Dataset.from_dict(
                dataset_buffer[
                    len(dataset_buffer)
                    // number_sample_per_shard
                    * number_sample_per_shard :
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
        state = write_shard(dataset_buffer, output_dir, output_format, state)
        state.save_state(state_path)

    if push_to_hub:
        if output_format == "jsonl" or output_format == "json":
            # jsonl without state.jsonl
            files = os.listdir(output_dir)
            if "state.jsonl" in files:
                files.remove("state.jsonl")
            # Sort files by shard id to keep the order.
            files.sort(key=lambda x: int(x.split(".")[0]))
            translated_ds = load_dataset(
                "json", data_files=[os.path.join(output_dir, f) for f in files]
            )
        elif output_format == "parquet":
            translated_ds = load_dataset(
                "parquet", data_files=os.path.join(output_dir, "*.parquet")
            )
        translated_ds.push_to_hub(push_to_hub_path, private=True)


if __name__ == "__main__":
    main()
