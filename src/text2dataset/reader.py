from datasets import load_dataset, IterableDataset
from text2dataset.utils import State


def create_dataset(input_path: str, state: State) -> IterableDataset:
    """
    Create a Iterabledataset from the input path.
    The input path can be a local file path or a HuggingFace Datasets dataset name.
    Use the state to skip already processed examples.
    """
    match input_path.split(".")[-1]:
        case "csv":
            ds = load_dataset("csv", data_files=input_path, streaming=True)
        case "json":
            ds = load_dataset("json", data_files=input_path, streaming=True)
        case "jsonl":
            ds = load_dataset("json", data_files=input_path, streaming=True)
        case "parquet":
            ds = load_dataset("parquet", data_files=input_path, streaming=True)
        case "tar":
            ds = load_dataset("webdataset", data_files=input_path, streaming=True)
        case "arrow":
            ds = load_dataset("arrow", data_files=input_path, streaming=True)
        case "txt":
            ds = load_dataset("text", data_files=input_path, streaming=True)
        case _:
            ds = load_dataset(input_path, streaming=True)

    # skip already processed examples
    if state.last_saved_example_num > 0:
        ds["train"] = ds["train"].skip(state.last_saved_example_num)

    return ds


def test_create_dataset():
    input_path = "Abirate/english_quotes"
    state = State(
        current_shard_id=0, last_saved_example_num=0, total_processed_examples=0
    )
    ds = create_dataset(input_path, state)
    assert ds["train"] is not None
    # error happing case
    try:
        create_dataset("hogehuga", state)
        # Unreachable Error
        assert False
    except Exception as e:
        assert e is not None


if __name__ == "__main__":
    input_path = "Abirate/english_quotes"
    state = State(
        current_shard_id=0, last_saved_example_num=0, total_processed_examples=0
    )
    ds = create_dataset(input_path, state)
    print(ds["train"])
    # iterabledatasetdict to datasetdict
    input_path = "data/english_quotes.json"
    ds = create_dataset(input_path, state)
    print(ds["train"])
    print(next(iter(ds["train"])))
    ds.map(lambda x: x["txt"], batched=True)
