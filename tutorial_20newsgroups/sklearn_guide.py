"""ref: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html"""

import argparse

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "classifier", choices=("MultinomialNB", "SGDClassifier")
    )
    args = parser.parse_args()

    if args.classifier == "MultinomialNB":
        clf = MultinomialNB()
    elif args.classifier == "SGDClassifier":
        clf = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-3,
            random_state=42,
            max_iter=5,
            tol=None,
        )

    categories = [
        "alt.atheism",
        "soc.religion.christian",
        "comp.graphics",
        "sci.med",
    ]
    twenty_train = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42
    )
    print(f"twenty_train: {len(twenty_train.data)}")

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    print(f"X_train_counts: {X_train_counts.shape}")

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(f"X_train_tfidf: {X_train_tfidf.shape}")

    clf.fit(X_train_tfidf, twenty_train.target)
    print(np.mean(clf.predict(X_train_tfidf) == twenty_train.target))

    twenty_test = fetch_20newsgroups(
        subset="test", categories=categories, shuffle=True, random_state=42
    )
    print(f"twenty_test: {len(twenty_test.data)}")
    X_test_counts = count_vect.transform(twenty_test.data)
    print(f"X_test_counts: {X_test_counts.shape}")
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print(f"X_test_tfidf: {X_test_tfidf.shape}")
    predicted = clf.predict(X_test_tfidf)
    print(np.mean(predicted == twenty_test.target))
