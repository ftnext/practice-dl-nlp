# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "openai",
# ]
# ///

# ref: https://huggingface.co/datasets/elyza/ELYZA-tasks-100/blob/main/baseline/scripts/gemini_pro.py

import os

from datasets import load_dataset, Dataset
from openai import OpenAI

if __name__ == "__main__":
    model_name = "gemini-2.0-flash-exp"

    elyza_tasks = load_dataset(
        "elyza/ELYZA-tasks-100", revision="1.0.0", split="test"
    )
    sample_tasks = Dataset.from_list([elyza_tasks[81], elyza_tasks[14]])

    client = OpenAI(
        api_key=os.environ["GOOGLE_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    )

    def pred(example):
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": example["input"]}],
            model=model_name,
            temperature=0.2,
            # top_p=0.8,  # OpenAIのAPIドキュメントにはtemperatureと同時に変えるべきではない
            # top_k=40,  # GoogleのAPI側だけの模様
            max_completion_tokens=200,
        )
        example[model_name] = response.choices[0].message.content
        return example

    sample_tasks = sample_tasks.map(pred, batched=False)
    sample_tasks.to_csv(f"preds/{model_name}.csv", index=False)
