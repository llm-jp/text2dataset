from vllm import LLM, SamplingParams


def build_prompt(passage: str) -> str:
    if passage is None:
        return ""
    passage = passage.strip().replace("\n", "")
    passage = f"""
You are an excellent English-Japanese translator. Please translate the following sentence into Japanese.
You must output only the translation.

Sentence: {passage}

Translation:
""".strip()

    return passage


class Translator:
    def __init__(
        self, model_id: str, tensor_parallel_size: int, pipeline_parallel_size: int
    ):
        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )

    def translate(self, text_list: list[str]) -> list[str]:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)
        prompts = [build_prompt(t) for t in text_list]
        outputs = self.llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        translated = [
            generated_text.replace(prompts[i], "").strip()
            for i, generated_text in enumerate(generated_texts)
        ]
        return translated


class MockTranslator:
    def __init__(
        self, model_id: str, tensor_parallel_size: int, pipeline_parallel_size: int
    ):
        pass

    def translate(self, text_list: list[str]) -> list[str]:
        return text_list


if __name__ == "__main__":
    model_id = "google/gemma-2-9b-it"
    tensor_parallel_size = 1
    pipeline_parallel_size = 1
    #translator = MockTranslator(model_id, tensor_parallel_size, pipeline_parallel_size)
    translator = Translator(model_id, tensor_parallel_size, pipeline_parallel_size)
    text_list = ["Hello, how are you?", "“Be yourself; everyone else is already taken.”"]
    translated = translator.translate(text_list)
    print(translated)
