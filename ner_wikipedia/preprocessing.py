from __future__ import annotations

import re
from typing import Optional

import numpy
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Vocab:
    """ボキャブラリの構築とそれを使った単語のID化（整数に変換）を行うクラス

    テキストを構成する単語と、対応するラベルをそれぞれボキャブラリーにして扱う
    """

    def __init__(self, num_words=None, lower=True, oov_token=None):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=num_words,
            oov_token=oov_token,
            filters="",
            lower=lower,
        )

    def fit(self, sequences: list[list[str]]) -> "Vocab":
        self.tokenizer.fit_on_texts(sequences)
        return self

    def encode(self, sequences: list[list[str]]) -> list[list[int]]:
        return self.tokenizer.texts_to_sequences(sequences)

    def decode(self, sequences: list[list[int]]) -> list[str]:
        texts = self.tokenizer.sequences_to_texts(sequences)
        return [text.split(" ") for text in texts]

    def _texts(self, sequences: list[list[str]]) -> list[str]:
        # fit_on_texts, texts_to_sequences ともに list[list[str]] を受け取れるので
        # このメソッドは不要。使わないなら、Tokenizerのsplit引数に"\t"を渡さなくてもよい
        return ["\t".join(words) for words in sequences]

    def get_index(self, word: str) -> Optional[int]:
        return self.tokenizer.word_index.get(word)

    @property
    def size(self) -> int:
        return len(self.tokenizer.word_index) + 1

    def save(self, file_path):
        with open(file_path, "w") as f:
            config = self.tokenizer.to_json()
            f.write(config)

    @classmethod
    def load(cls, file_path):
        with open(file_path) as f:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                f.read()
            )
            vocab = cls()
            vocab.tokenizer = tokenizer
        return vocab


def normalize_number(text: str, reduce: bool = True) -> str:
    """
    >>> text = "2万0689・24ドル"
    >>> normalize_number(text)
    '0万0・0ドル'
    >>> normalize_number(text, reduce=False)
    '0万0000・00ドル'
    """
    if reduce:
        normalized_text = re.sub(r"\d+", "0", text)
    else:
        normalized_text = re.sub(r"\d", "0", text)
    return normalized_text


def preprocess_dataset(sequences: list[list[str]]) -> list[list[str]]:
    """各文字列について、数字の並び部分を0に正規化する

    >>> from utils import load_dataset
    >>> sentences, _ = load_dataset("data/ja.wikipedia.conll")
    >>> preprocessed = preprocess_dataset(sentences)
    >>> sentences[0]  # doctest: +SKIP
    ['1960', '年代', 'と', '1970', '年代', 'の', '間', 'に', ...]
    >>> preprocessed[0]  # doctest: +SKIP
    ['0', '年代', 'と', '0', '年代', 'の', '間', 'に', ...]
    """
    return [[normalize_number(w) for w in words] for words in sequences]


def create_dataset(
    sequences: list[list[str]], vocab: Vocab
) -> "numpy.ndarray":
    encoded_sequences = vocab.encode(sequences)
    return pad_sequences(encoded_sequences, padding="post")
