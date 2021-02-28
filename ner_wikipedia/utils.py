from __future__ import annotations


def load_dataset(
    filename: str, encoding="utf-8"
) -> tuple(list[list[str]], list[list[str]]):
    """データセットを読み込むための関数

    データセットの1行：単語\t固有表現タイプ

    データセットには空行を含む。空行により、複数の文のまとまりが区切られる

    >>> sentences, labels = load_dataset("data/ja.wikipedia.conll")
    >>> sentences[0]  # doctest: +SKIP
    ['1960', '年代', 'と', '1970', '年代', 'の', '間', 'に', ...]
    >>> labels[0]  # 対応する固有表現タイプ  # doctest: +SKIP
    ['B-DATE', 'I-DATE', 'O', 'B-DATE', 'I-DATE', 'O', 'O', 'O', ...]
    """
    sentences, labels = [], []
    words, tags = [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split("\t")
                words.append(word)
                tags.append(tag)
            else:  # 文書の区切りの空行
                sentences.append(words)
                labels.append(tags)
                words, tags = [], []
        if words:  # 最後の文書の処理
            sentences.append(words)
            labels.append(tags)

    return sentences, labels
