import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],  # suffix
        "word[-2:]": word[-2:],  # suffix
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],  # 品詞を表す文字列の先頭2文字
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:postag": postag1,
                "-1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:postag": postag1,
                "+1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    """
    >>> train_sent = [
    ...   ("Melbourne", "NP", "B-LOC"),
    ...   ("(", "Fpa", "O"),
    ...   ("Australia", "NP", "B-LOC"),
    ...   (")", "Fpt", "O"),
    ...   (",", "Fc", "O"),
    ...   ("25", "Z", "O"),
    ...   ("may", "NC", "O"),
    ...   ("(", "Fpa", "O"),
    ...   ("EFE", "NC", "B-ORG"),
    ...   (")", "Fpt", "O"),
    ...   (".", "Fp", "O"),
    ... ]
    >>> actual = sent2features(train_sent)[0]
    >>> expected = {
    ...   "+1:postag": "Fpa",
    ...   "+1:postag[:2]": "Fp",
    ...   "+1:word.istitle()": False,
    ...   "+1:word.isupper()": False,
    ...   "+1:word.lower()": "(",
    ...   "BOS": True,
    ...   "bias": 1.0,
    ...   "postag": "NP",
    ...   "postag[:2]": "NP",
    ...   "word.isdigit()": False,
    ...   "word.istitle()": True,
    ...   "word.isupper()": False,
    ...   "word.lower()": "melbourne",
    ...   "word[-2:]": "ne",
    ...   "word[-3:]": "rne",
    ... }
    >>> actual == expected
    True
    """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


if __name__ == "__main__":
    train_sents = nltk.corpus.conll2002.iob_sents("esp.train")
    test_sents = nltk.corpus.conll2002.iob_sents("esp.testb")

    X_train, y_train = [], []
    for s in train_sents:
        X_train.append(sent2features(s))
        y_train.append(sent2labels(s))
    print("Train:", len(X_train), len(y_train))

    X_test, y_test = [], []
    for s in test_sents:
        X_test.append(sent2features(s))
        y_test.append(sent2labels(s))
    print("Test:", len(X_test), len(y_test))

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove("O")  # 評価に使うラベルはO以外

    y_pred = crf.predict(X_test)
    f1_score = metrics.flat_f1_score(
        y_test, y_pred, average="weighted", labels=labels
    )
    print(f"{f1_score=}")

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(
        metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        )
    )
