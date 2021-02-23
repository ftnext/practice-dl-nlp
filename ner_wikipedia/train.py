import argparse

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from seqeval.metrics import classification_report

from inference import InferenceAPI
from models import BidirectionalModel, UnidirectionalModel
from preprocessing import create_dataset, preprocess_dataset, Vocab
from utils import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type", choices=("unidirectional", "bidirectional")
    )
    args = parser.parse_args()

    batch_size = 32
    epochs = 100
    model_path = f"models/{args.model_type}_model.h5"
    num_words = 15000

    model_class = BidirectionalModel
    if args.model_type == "unidirectional":
        model_class = UnidirectionalModel

    x, y = load_dataset("./data/ja.wikipedia.conll")

    x = preprocess_dataset(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    source_vocab = Vocab(num_words=num_words, oov_token="<UNK>").fit(x_train)
    target_vocab = Vocab(lower=False).fit(y_train)
    x_train = create_dataset(x_train, source_vocab)
    y_train = create_dataset(y_train, target_vocab)

    model = model_class(num_words, target_vocab.size).build()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(model_path, save_best_only=True),
    ]

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        shuffle=True,
    )

    model = load_model(model_path)
    api = InferenceAPI(model, source_vocab, target_vocab)
    y_pred = api.predict_from_sequences(x_test)
    print(classification_report(y_test, y_pred, digits=4))
