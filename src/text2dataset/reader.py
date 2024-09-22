from datasets import load_dataset, DatasetDict
from text2dataset.utils import State


def create_dataset(input_format: str, input_path: str, state: State) -> DatasetDict:
    ds = load_dataset(input_format, data_files={input_path}, streaming=True)
    # skip already processed examples
    if state.last_saved_example_num > 0:
        ds["train"] = ds["train"].skip(state.last_saved_example_num)

    return ds
