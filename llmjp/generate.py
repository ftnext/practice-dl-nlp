import argparse
import time

import torch
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)


def main(model, tokenizer):
    while True:
        input_text = input("日本語テキストを入力してください(qで終了): ")
        input_text = input_text.rstrip()
        if not input_text:
            continue
        if input_text.lower() == "q":
            break

        output = generate(model, tokenizer, input_text)
        print(
            f"{Fore.YELLOW}{input_text}{Fore.WHITE}{output[len(input_text):]}"
        )


def generate(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    print("----- generate start -----")
    with torch.no_grad():
        start = time.time()
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        print(time.time() - start)
    print("----- generate end -----")
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    main(model, tokenizer)
