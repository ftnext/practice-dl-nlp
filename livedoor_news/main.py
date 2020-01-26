import os

from janome.analyzer import Analyzer
from janome.tokenfilter import (
    CompoundNounFilter,
    ExtractAttributeFilter,
    LowerCaseFilter,
    POSStopFilter,
)
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(1234)


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


def janome_tokenizer():
    token_filters = [
        CompoundNounFilter(),
        POSStopFilter(["記号", "助詞", "助動詞", "名詞,非自立", "名詞,代名詞"]),
        LowerCaseFilter(),
        ExtractAttributeFilter("surface"),
    ]
    tokenizer = Tokenizer()
    analyzer = Analyzer(tokenizer=tokenizer, token_filters=token_filters)
    analyze_function = analyzer.analyze

    def _tokenizer(text_tensor, label):
        text_str = text_tensor.numpy().decode("utf-8")
        tokenized_text = " ".join(list(analyze_function(text_str)))
        return tokenized_text, label

    return _tokenizer


def tokenize_map_fn(tokenizer):
    def _tokenize_map_fn(text_tensor, label):
        return tf.py_function(
            tokenizer, inp=[text_tensor, label], Tout=(tf.string, tf.int64)
        )

    return _tokenize_map_fn


def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    encoded_text, label = tf.py_function(
        encode, inp=[text, label], Tout=(tf.int64, tf.int64)
    )
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label


def plot_accuracy(history):
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, "bo", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.legend()
    plt.savefig("accuracy.png")


BUFFER_SIZE = 2000
BATCH_SIZE = 64
TAKE_SIZE = 150

text_datasets = []
label_count = 0

text_dir = os.path.join(os.getcwd(), "text")
text_subdir_names = tf.compat.v1.gfile.ListDirectory(text_dir)
for subdir in text_subdir_names:
    data_dir = os.path.join(text_dir, subdir)
    if os.path.isdir(data_dir):
        print(f"{label_count}: {subdir}")
        text_file_names = tf.compat.v1.gfile.ListDirectory(data_dir)
        text_tensors = []
        for file_name in text_file_names:
            text_file = os.path.join(data_dir, file_name)
            lines_dataset = tf.data.TextLineDataset(text_file)
            # 1行1行がTensorとなるので、ファイルの文章全体をつないでTensorとする
            sentences = [
                line_tensor.numpy().decode("utf-8")
                for line_tensor in lines_dataset
            ]
            concatenated_sentences = " ".join(sentences)
            # subdirのファイルごとにTensorを作り、Datasetとする
            text_tensor = tf.convert_to_tensor(concatenated_sentences)
            text_tensors.append(text_tensor)
        text_dataset = tf.data.Dataset.from_tensor_slices(text_tensors)
        text_dataset = text_dataset.map(lambda ex: labeler(ex, label_count))
        text_datasets.append(text_dataset)
        label_count += 1

print("text data loaded")
all_labeled_data = text_datasets[0]
for labeled_data in text_datasets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_data)

all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, seed=RANDOM_SEED, reshuffle_each_iteration=False
)
print("text data shuffled")

all_tokenized_data = all_labeled_data.map(tokenize_map_fn(janome_tokenizer()))
print("japanese text tokenized by janome")

tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
# 評価しているから時間がかかる
for text_tensor, _ in tqdm(all_tokenized_data):
    tokens = tokenizer.tokenize(text_tensor.numpy().decode("utf-8"))
    vocabulary_set.update(tokens)
vocab_size = len(vocabulary_set)
print(f"vocabulary size: {vocab_size}")

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
all_encoded_data = all_tokenized_data.map(encode_map_fn)
print("text encoded")

output_shapes = tf.compat.v1.data.get_output_shapes(all_encoded_data)
test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, output_shapes)
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(
    BUFFER_SIZE, seed=RANDOM_SEED
)

val_data = train_data.take(TAKE_SIZE)
val_data = val_data.padded_batch(BATCH_SIZE, output_shapes)
train_data = train_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE, seed=RANDOM_SEED)

train_data = train_data.padded_batch(BATCH_SIZE, output_shapes)
vocab_size += 1  # padded_batchした際に0という番号を追加している
print("data splitted. Start train")


model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_data, epochs=3, validation_data=val_data)

eval_loss, eval_acc = model.evaluate(test_data)
print(f"On test data: loss={eval_loss}, acc={eval_acc}")
plot_accuracy(history.history)
