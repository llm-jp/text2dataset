# text2dataset
[![pypi](https://img.shields.io/pypi/v/text2dataset.svg)](https://pypi.python.org/pypi/text2dataset)

Easily turn large English text datasets into Japanese text datasets using open LLMs.

<figure>
  <img src="images/english_quotes_ja.png" width="50%">
  <figcaption> Fig: Japanese translation of the <a href="https://huggingface.co/datasets/Abirate/english_quotes">Abirate/english_quotes</a> dataset using the  <a href="https://huggingface.co/llm-jp/llm-jp-3-3.7b-instruct">llm-jp/llm-jp-3-3.7b-instruct</a> model. </figcaption>
</figure>

## Overview
text2dataset is a tool for converting a datasets.Dataset by translating the data in the "txt" column using Open LLM like gemma2 with vLLM, and adding a new "txt_ja" column (translated text in Japanese). You can also use text2dataset to paraphrase texts by changing the prompt template.
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
- Custom Prompt Template:
  - By specifying the `prompt_template_path` parameter, you can customize the prompt template for any translation task (e.g., paraphrasing, summarization etc.).

## Installation
```bash
$ git clone https://github.com/llm-jp/text2dataset.git
$ rye sync
```

## Usage

### Translation
```bash
$ python src/text2dataset/main.py \
    --model_id llm-jp/llm-jp-3-3.7b-instruct \
    --batch_size 16384 \
    --input_path data/english_quotes.json \
    --source_column text \
    --target_column text_ja \
    --push_to_hub True \
    --push_to_hub_path speed/english_quotes_ja \
    --output_dir data/english_quotes_ja \
    --output_format json
```

Using the [`llm-jp/llm-jp-3-3.7b-instruct`](https://huggingface.co/llm-jp/llm-jp-3-3.7b-instruct) model on an A100 GPU, 2508 English quotes were translated into Japanese in just 21 seconds.

<figure>
  <img src="images/english_quotes_ja.png" width="50%">
  <figcaption> Fig: Japanese translation of the <a href="https://huggingface.co/datasets/Abirate/english_quotes">Abirate/english_quotes</a> dataset using the  <a href="https://huggingface.co/llm-jp/llm-jp-3-3.7b-instruct">llm-jp/llm-jp-3-3.7b-instruct</a> model. </figcaption>
</figure>

The result dataset is available at [speed/english_quotes_ja](https://huggingface.co/datasets/speed/english_quotes).

### Paraphrasing
You can also use text2dataset to paraphrase texts by changing the prompt template with specifying the `prompt_template_path` parameter.
```bash
$ python src/text2dataset/main.py \
    --model_id google/gemma-2-2b-it \
    --batch_size 16384 \
    --input_path data/english_quotes.json \
    --source_column text \
    --target_column text_paraphrase \
    --push_to_hub True \
    --push_to_hub_path speed/english_quotes_paraphrase \
    --output_dir data/english_quotes_paraphrase \
    --output_format json \
    --prompt_template_path config/paraphrase.yaml
```

<figure>
  <img src="images/english_quotes_paraphrase.png" width="50%">
  <figcaption> Fig: Paraphrase of the <a href="https://huggingface.co/datasets/Abirate/english_quotes">Abirate/english_quotes</a> dataset using the  <a href="https://huggingface.co/google/gemma-2-2b-it">google/gemma-2-2b-it/</a> model. </figcaption>
</figure>

The result dataset is available at [speed/english_quotes_paraphrase](https://huggingface.co/datasets/speed/english_quotes_paraphrase).


### Translation of [neuralwork/arxiver](https://huggingface.co/datasets/neuralwork/arxiver) dataset
You can directly translate datasets in Hugging Face by specifying the path name in `input_path`.

In this example, the `abstract` column of the `neuralwork/arxiver` dataset is translated by specifying the `input_path` as `neuralwork/arxiver` and the `source_column` parameter as `abstract`.
```bash
$ python src/text2dataset/main.py \
    --model_id google/gemma-2-2b-it \
    --batch_size 16384 \
    --input_path neuralwork/arxiver \
    --source_column abstract \
    --target_column abstract_ja \
    --push_to_hub True \
    --push_to_hub_path speed/arxiver_ja \
    --output_dir data/arxiver_ja \
    --output_format json \
    --use_wandb True \
    --wandb_run_name arxiver
```

`neuralwork/arxiver` dataset contains 138k rows of abstracts, and it took 2.5 hours to translate them into Japanese using the `google/gemma-2-2b-it` model on a A100 GPU. The result dataset is available at [speed/arxiver_ja](https://huggingface.co/datasets/speed/arxiver_ja).

<figure>
  <img src="images/arxiver_ja.png" width="50%">
  <figcaption> Fig: Translation of the <a href="https://huggingface.co/datasets/neuralwork/arxiver">neuralwork/arxiver</a> dataset using the  <a href="https://huggingface.co/google/gemma-2-2b-it">google/gemma-2-2b-it/</a> model. </figcaption>
</figure>


<figure>
  <img src="images/arxiver_wandb.png" width="50%">
  <figcaption> Fig: Wandb logs for the translation of the <a href="https://huggingface.co/datasets/neuralwork/arxiver">neuralwork/arxiver</a> dataset using the  <a href="https://huggingface.co/google/gemma-2-2b-it">google/gemma-2-2b-it/</a> model. </figcaption>
</figure>



## Tips

- Translation on Multiple GPUs in Parallel

To run translations on multiple GPUs concurrently, split the input dataset into several shards (directories) and execute the translation for each shard in parallel. Remember to set the gpu_id parameter to the corresponding GPU ID for each shard.


## Areas for Improvement

### Data Parallel Inference

Currently, we need to manually split the input dataset into shards and run the translation for each shard in parallel to utilize multiple GPUs. It would be great to have a built-in feature to automatically split the input dataset into shards and run the translation on multiple GPUs in parallel.
If you have any ideas or suggestions, please feel free to open an issue or Pull Request.

## Development

### Contribution
Welcome to any contributions!
If you have any questions or suggestions, please feel free to open an issue or Pull Request.

### PyPI Release
```bash
git tag -a v0.x.x -m "version 0.x.x"
git push origin --tags
```
### Lint and Format
```bash
$ rye lint
$ rye format
```


## References
- https://github.com/vllm-project/vllm
- https://github.com/rom1504/img2dataset
- https://huggingface.co/datasets/Abirate/english_quotes
