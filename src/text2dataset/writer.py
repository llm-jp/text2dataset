import os
from datasets import Dataset
import logging
from text2dataset.utils import State

logger = logging.getLogger(__name__)


def write_shard(
    dataset: Dataset, output_dir: str, output_format: str, state: State
) -> State:
    """Write a shard of the dataset to a file.
    Supported output formats are "jsonl" and "parquet".
    """
    if output_format == "jsonl" or output_format == "json":
        output_path = os.path.join(output_dir, f"{state.current_shard_id:05d}.jsonl")
        dataset.to_json(output_path, force_ascii=False, lines=True)
        abs_path = os.path.abspath(output_path)
        logger.info(f"Shard {state.current_shard_id} written to {abs_path}")
    elif output_format == "parquet":
        output_path = os.path.join(output_dir, f"{state.current_shard_id:05d}.parquet")
        dataset.to_parquet(output_path)
        abs_path = os.path.abspath(output_path)
        logger.info(f"Shard {state.current_shard_id} written to {abs_path}")
    else:
        raise ValueError(f"Unknown output format: {output_format}")
    state.last_saved_example_num += len(dataset)
    return state
