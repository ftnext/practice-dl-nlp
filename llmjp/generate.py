import argparse
import logging
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
    logger = logging.getLogger(__name__)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    logger.debug("generate start")
    with torch.no_grad():
        start = time.time()
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        logger.debug(time.time() - start)
    logger.debug("generate end")
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        log_format = (
            "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d"
            " - %(message)s"
        )
        logging.basicConfig(level=logging.DEBUG, format=log_format)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    main(model, tokenizer)
