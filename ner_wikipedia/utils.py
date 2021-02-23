def load_dataset(filename, encoding="utf-8"):
    """データセットを読み込むための関数

    データセットの1行：単語\t固有表現タイプ

    データセットには空行を含む。空行により、複数の文のまとまりが区切られる

    >>> sentences, labels = load_dataset("data/ja.wikipedia.conll")
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
