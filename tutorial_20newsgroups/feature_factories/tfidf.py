import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

CATEGORIES = [
    "alt.atheism",
    "soc.religion.christian",
    "comp.graphics",
    "sci.med",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root_path", type=Path)
    args = parser.parse_args()

    twenty_train = fetch_20newsgroups(
        subset="train", categories=CATEGORIES, shuffle=True, random_state=42
    )
    print(f"twenty_train: {len(twenty_train.data)}")
    tfidf_pipe = Pipeline(
        [("vect", CountVectorizer()), ("tfidf", TfidfTransformer())]
    )
    X_train_tfidf = tfidf_pipe.fit_transform(twenty_train.data)
    print(f"X_train_tfidf: {X_train_tfidf.shape}")

    twenty_test = fetch_20newsgroups(
        subset="test", categories=CATEGORIES, shuffle=True, random_state=42
    )
    X_test_tfidf = tfidf_pipe.transform(twenty_test.data)
    print(f"X_test_tfidf: {X_test_tfidf.shape}")

    args.output_root_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_root_path / "train.npz",
        X=X_train_tfidf.toarray(),
        y=twenty_train.target,
    )
    np.savez(
        args.output_root_path / "test.npz",
        X=X_test_tfidf.toarray(),
        y=twenty_test.target,
    )
    with open(args.output_root_path / "feature_pipeline.pkl", "wb") as fb:
        pickle.dump(tfidf_pipe, fb)
