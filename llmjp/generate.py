import argparse
import logging
import time

import torch
from colorama import Fore, init
from transformers import AutoModelForCausalLM, AutoTokenizer
from yaspin import yaspin

init(autoreset=True)

logger = logging.getLogger(__name__)


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


def _stockmark_generate_impl(model, inputs):
    with torch.no_grad():
        tokens = model.generate(
            **inputs, max_new_tokens=128, repetition_penalty=1.1
        )
    return tokens


@yaspin(text="generating ...")
def _generate_impl(model, inputs):
    if model.name_or_path.startswith("stockmark"):
        return _stockmark_generate_impl(model, inputs)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokens


def generate(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start = time.time()
    tokens = _generate_impl(model, inputs)
    logger.debug("took: %f (sec)", time.time() - start)

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return output


def setup_root_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = (
        "%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d"
        " - %(message)s"
    )
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    module_log_pass_filter = logging.Filter(__name__)
    handler.addFilter(module_log_pass_filter)

    root_logger.addHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        setup_root_logger()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    main(model, tokenizer)
