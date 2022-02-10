import argparse
from pathlib import Path

import numpy as np
from fairseq.models.roberta import RobertaModel
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

CATEGORIES = [
    "alt.atheism",
    "soc.religion.christian",
    "comp.graphics",
    "sci.med",
]


def extract_features(data2vec, data):
    X_features = []
    for text in tqdm(data):
        tokens = data2vec.encode(text)
        last_layer_features = data2vec.extract_features(tokens[:512])
        feature_array = last_layer_features[0][0].detach().numpy()
        X_features.append(feature_array)
    return np.array(X_features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_dir_path", help="Directory path. dict.txt is needed."
    )
    parser.add_argument("output_root_path", type=Path)
    parser.add_argument("--model_name", default="nlp_base.pt")
    args = parser.parse_args()

    data2vec = RobertaModel.from_pretrained(
        args.model_dir_path, args.model_name
    )
    data2vec.eval()

    twenty_train = fetch_20newsgroups(
        subset="train", categories=CATEGORIES, shuffle=True, random_state=42
    )
    print(f"twenty_train: {len(twenty_train.data)}")
    X_train = extract_features(data2vec, twenty_train.data)
    print(f"X_train: {X_train.shape}")

    twenty_test = fetch_20newsgroups(
        subset="test", categories=CATEGORIES, shuffle=True, random_state=42
    )
    print(f"twenty_test: {len(twenty_test.data)}")
    X_test = extract_features(data2vec, twenty_test.data)
    print(f"X_test: {X_test.shape}")

    args.output_root_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_root_path / "train.npz",
        X=X_train,
        y=twenty_train.target,
    )
    np.savez(
        args.output_root_path / "test.npz",
        X=X_test,
        y=twenty_test.target,
    )
