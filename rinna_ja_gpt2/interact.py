import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, T5Tokenizer


def freeze_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


model_name = "rinna/japanese-gpt2-medium"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

length = 100
temperature = 1.0
k = 0
p = 0.9
repetition_penalty = 1.0
num_return_sequences = 3

while True:
    input_text = input("日本語テキストを入力してください(qで終了): ")
    input_text = input_text.rstrip()
    if not input_text:
        continue
    if input_text.lower() == "q":
        break

    freeze_seed()
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_text),
        temperature=temperature,
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    generated_sequences = []
    for idx, sequence in enumerate(output_sequences):
        print(f"=== GENERATED SEQUENCE {idx + 1} ===")
        sequence = sequence.tolist()  # tensor -> list

        text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)

        # textの先頭の {input_text}</s> を除く？
        total_sequence = (
            input_text
            + text[
                len(
                    tokenizer.decode(
                        input_ids[0], clean_up_tokenization_spaces=True
                    )
                ) :
            ]
        )

        print(total_sequence)
        generated_sequences.append(total_sequence)
