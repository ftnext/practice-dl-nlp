"""ref: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_feature_path", type=Path)
    parser.add_argument("test_feature_path", type=Path)
    parser.add_argument("output_root_path", type=Path)
    parser.add_argument(
        "--classifier",
        "-c",
        choices=("MultinomialNB", "SGDClassifier"),
        nargs="+",
        required=True,
    )
    args = parser.parse_args()

    args.output_root_path.mkdir(parents=True, exist_ok=True)

    classifiers = []
    if "MultinomialNB" in args.classifier:
        classifiers.append(MultinomialNB())
    if "SGDClassifier" in args.classifier:
        classifiers.append(
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=1e-3,
                random_state=42,
                max_iter=5,
                tol=None,
            )
        )

    train = np.load(args.train_feature_path)
    test = np.load(args.test_feature_path)

    for name, clf in zip(args.classifier, classifiers):
        print(f"train {name}")
        clf.fit(train["X"], train["y"])

        train_predicted = clf.predict(train["X"])
        print(np.mean(train_predicted == train["y"]))
        test_predicted = clf.predict(test["X"])
        print(np.mean(test_predicted == test["y"]))

        with open(args.output_root_path / f"{name}.pkl", "wb") as fb:
            pickle.dump(clf, fb)
