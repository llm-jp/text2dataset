# text2dataset
[![pypi](https://img.shields.io/pypi/v/text2dataset.svg)](https://pypi.python.org/pypi/text2dataset)

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
You can use Translator class to translate texts into Japanese.
```python
from datasets import load_dataset
from text2dataset.translator import Translator

ds = load_dataset("Abirate/english_quotes", split="train")
ds = ds.select(range(10))
print(ds.column_names)
# ['quote', 'author', 'tags']
print("\n".join(ds["quote"][:5]))
# “Be yourself; everyone else is already taken.”
# “I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.”
# “Two things are infinite: the universe and human stupidity; and I'm not sure about the universe.”
# “So many books, so little time.”
# “A room without books is like a body without a soul.”
translator = Translator(model_id="google/gemma-2-9b-it", tensor_parallel_size=1, pipeline_parallel_size=1)
translated = translator.translate(ds["quote"])
ds = ds.add_column("quote_ja", translated)
print(ds.column_names)
# ['quote', 'author', 'tags', 'quote_ja']
print("\n".join(ds["quote_ja"][:5]))
#
# 自分のことは、自己中心的で、衝動的で、少し不安定。失敗することもあるし、制御不能な時もあるし、扱いにくい時もある。でも、私が最悪な時をあなたが処理できないなら、最高の私をあなたが望む資格はない。
# **宇宙と人間の愚かさ、どちらが無限大か分からない。**
# 本がたくさん、時間が足りない。
# 書籍のない部屋は、魂のない体と同じ。
```

## Areas for Improvement
- Data Paarallel Inference:
  - Currently, only one model is used for inference. This can be improved by using DataParallel. If you know how to do this with vLLM, please let me know or Pull Request.




## References
- https://github.com/vllm-project/vllm
- https://github.com/rom1504/img2dataset
