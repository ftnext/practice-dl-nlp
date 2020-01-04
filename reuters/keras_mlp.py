import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer


np.random.seed(42)
tf.random.set_seed(1234)

MAX_WORDS = 1000
DROPOUT = 0.5
OPTIMIZER = keras.optimizers.Adam()
BATCH_SIZE = 32
EPOCHS = 5


class IndexWordMapper:
    def __init__(self, index_word_map):
        self.index_word_map = index_word_map

    @staticmethod
    def initialize_index_word_map():
        word_index = reuters.get_word_index()
        index_word_map = {
            index + 3: word for word, index in word_index.items()
        }
        index_word_map[0] = "[padding]"
        index_word_map[1] = "[start]"
        index_word_map[2] = "[oov]"
        return index_word_map

    def print_original_sentence(self, indices_of_words):
        for index in indices_of_words:
            print(self.index_word_map[index], end=" ")


class TokenizePreprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def initialize_tokenizer(max_words):
        return Tokenizer(num_words=max_words)

    def convert_text_to_matrix(self, texts, mode):
        return self.tokenizer.sequences_to_matrix(texts, mode=mode)


def convert_to_onehot(labels, number_of_classes):
    return keras.utils.to_categorical(labels, number_of_classes)


def build_model(number_of_classes, max_words, drop_out, optimizer):
    model = keras.Sequential(
        [
            layers.Dense(512, input_shape=(max_words,), activation=tf.nn.relu),
            layers.Dropout(drop_out),
            layers.Dense(number_of_classes, activation=tf.nn.softmax),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model


def plot_accuracy(history):
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.savefig("accuracy.png")


if __name__ == "__main__":
    index_word_map = IndexWordMapper.initialize_index_word_map()
    index_word_mapper = IndexWordMapper(index_word_map)

    (x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=MAX_WORDS
    )
    number_of_classes = np.max(y_train) + 1

    tokenizer = TokenizePreprocessor.initialize_tokenizer(MAX_WORDS)
    preprocessor = TokenizePreprocessor(tokenizer)
    x_train = preprocessor.convert_text_to_matrix(x_train, "binary")
    x_test = preprocessor.convert_text_to_matrix(x_test, "binary")

    y_train = convert_to_onehot(y_train, number_of_classes)
    y_test = convert_to_onehot(y_test, number_of_classes)

    model = build_model(number_of_classes, MAX_WORDS, DROPOUT, OPTIMIZER)

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_split=0.1,
    )
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print(score)

    plot_accuracy(history.history)
