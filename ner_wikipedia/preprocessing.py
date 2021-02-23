import re

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
            split="\t",
        )

    def fit(self, sequences):
        texts = self._texts(sequences)
        self.tokenizer.fit_on_texts(texts)
        return self

    def encode(self, sequences):
        texts = self._texts(sequences)
        return self.tokenizer.texts_to_sequences(texts)

    def decode(self, sequences):
        texts = self.tokenizer.sequences_to_texts(sequences)
        return [text.split(" ") for text in texts]

    def _texts(self, sequences):
        return ["\t".join(words) for words in sequences]

    def get_index(self, word):
        return self.tokenizer.word_index.get(word)

    @property
    def size(self):
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


def normalize_number(text, reduce=True):
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


def preprocess_dataset(sequences):
    return [[normalize_number(w) for w in words] for words in sequences]


def create_dataset(sequences, vocab):
    encoded_sequences = vocab.encode(sequences)
    return pad_sequences(encoded_sequences, padding="post")
