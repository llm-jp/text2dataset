from vllm import LLM, SamplingParams


def build_prompt(template: str, passage: str | None) -> str:
    if passage is None:
        return ""
    passage = passage.strip().replace("\n", "")
    return template.format(passage=passage).strip()


def test_build_prompt():
    template = "Translate the following text: {passage}"
    passage = "Hello, how are you?"
    prompt = build_prompt(template, passage)
    assert prompt == "Translate the following text: Hello, how are you?"


class Translator:
    """Translator class obtains a vLLM's LLM engine internally."""

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        template: str = "You are an excellent English-Japanese translator. Please translate the following sentence into Japanese.\nYou must output only the translation.\nSentence: {passage}\nTranslation:",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 200,
    ):
        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
        )
        self.template = template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def translate(self, text_list: list[str]) -> list[str]:
        sampling_params = SamplingParams(
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens
        )
        prompts = [build_prompt(self.template, t) for t in text_list]
        outputs = self.llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        translated = [
            generated_text.replace(prompts[i], "").strip()
            for i, generated_text in enumerate(generated_texts)
        ]
        return translated


class DeeplTranslator:
    """Translator class for DeepL API."""

    def __init__(self, target_lang: str = "JA"):
        import deepl
        from dotenv import load_dotenv
        import os

        load_dotenv()
        auth_key = os.getenv("DEEPL_API_KEY")
        self.translator = deepl.Translator(auth_key)
        self.target_lang = target_lang

    def translate(self, text_list: list[str]) -> list[str]:
        translated = []
        for text in text_list:
            result = self.translator.translate_text(text, target_lang=self.target_lang)
            translated.append(result.text)
        return translated


class OpenAIAPITranslator:
    """Translator class for OpenAI API."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini-2024-07-18",
        template: str = "You are an excellent English-Japanese translator. Please translate the following sentence into Japanese.",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 200,
    ):
        import os
        from openai import AzureOpenAI
        from dotenv import load_dotenv

        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.model_id = model_id
        self.template = template
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def translate(self, text_list: list[str]) -> list[str]:
        translated = []
        for text in text_list:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": self.template,
                    },
                    {"role": "user", "content": text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=0,
            )
            translated.append(response.choices[0].message.content)
        return translated


if __name__ == "__main__":
    text_list = [
        "Hello, how are you?",
        "“Be yourself; everyone else is already taken.”",
    ]

    model_id = "llm-jp/llm-jp-3-3.7b-instruct"
    tensor_parallel_size = 1
    pipeline_parallel_size = 1
    # translator = MockTranslator(model_id, tensor_parallel_size, pipeline_parallel_size)
    translator = Translator(model_id, tensor_parallel_size, pipeline_parallel_size)
    translated = translator.translate(text_list)
    print(translated)

    translator = DeeplTranslator()
    translated = translator.translate(text_list)
    print(translated)

    translator = OpenAIAPITranslator()
    translated = translator.translate(text_list)
    print(translated)
