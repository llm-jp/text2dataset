# text2dataset
Easily turn large English text datasets into Japanese text datasets using open LLMs.

A tool for converting a datasets.Dataset by translating the data in the "txt" column using Open LLM like gemma2 with vLLM, and adding a new "txt_ja" column (translated text in Japanese).
This tool is inspired by [img2dataset](https://github.com/rom1504/img2dataset).

## Features
- Save the intermediate results in shards:
  - By setting the `number_sample_per_shard` parameter, the dataset can be saved in shards as specified by the number of samples per shard.
- Resume from checkpoint:
  - By setting the `resume_from_checkpoint` parameter, the translation can be resumed from where it left off.
- Logging with wandb:
  - By setting the `use_wandb` parameter, the metrics such as examples_per_sec and count can be logged to wandb.
- Push to Hugging Face Hub:
  - By setting the `push_to_hub` parameter, the translated dataset can be pushed to the Hugging Face Hub.


## Usage

```bash
$ python src/text2dataset/main.py \
    --model_id "google/gemma-2-9b-it" \
    --batch_size 16384 \
    --input_format parquet \
    --input_path "/path/to/input" \
    --source_column "caption" \
    --target_column "caption_ja" \
    --push_to_hub False \
    --push_to_hub_path "/path/to/hub" \
    --output_dir "/path/to/output" \
    --output_format parquet \
    --gpu_id 0 \
    --number_sample_per_shard 10000 \
    --use_wandb True
```

### Example
```python
>>> from datasets import load_dataset
>>> load_dataset("parquet", data_files="/path/to/input", split="train")
DatasetDict({
    train: Dataset({
        features: ['__key__', '__url__', 'jpg', 'json', 'txt'],
        num_rows: 1000
    })
})
>>> load_dataset("parquet", data_files="/path/to/output")
DatasetDict({
    train: Dataset({
        features: ['__key__', '__url__', 'jpg', 'json', 'txt', 'txt_ja'],
        num_rows: 1000
    })
})
```

## Areas for Improvement
- Data Paarallel Inference:
  - Currently, only one model is used for inference. This can be improved by using DataParallel. If you know how to do this with vLLM, please let me know or Pull Request.




## References
- https://github.com/vllm-project/vllm
- https://github.com/rom1504/img2dataset
