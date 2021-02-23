import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class InferenceAPI:
    """学習したモデルを使ってラベルを予測するためのクラス"""

    def __init__(self, model, source_vocab, target_vocab):
        self.model = model
        self.source_vocab = source_vocab  # 単語のボキャブラリー
        self.target_vocab = target_vocab  # ラベルのボキャブラリー

    def predict_from_sequences(self, sequences):
        """入力されたテキストについて、系列のラベルを予測する

        テキストは単語リストのリストとして入力する
        [["1960", "年代", "と", ...], ...]

        出力は系列ラベルリストのリスト（入力に対応）
        """
        lengths = map(len, sequences)
        sequences = self.source_vocab.encode(sequences)
        sequences = pad_sequences(sequences, padding="post")
        y_pred = self.model.predict(sequences)
        y_pred = np.argmax(y_pred, axis=-1)  # 2次なのでaxis=1と同じ（ラベルのインデックスを取り出す）
        y_pred = self.target_vocab.decode(y_pred)
        # y[:l]で、postに追加したpadding部分を無視する
        return [y[:l] for y, l in zip(y_pred, lengths)]
