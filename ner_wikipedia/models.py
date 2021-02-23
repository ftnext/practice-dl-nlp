from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Embedding,
    Input,
    LSTM,
)


class UnidirectionalModel:
    """LSTMを使って、固有表現認識モデルを作成するためのクラス"""

    def __init__(self, input_dim, output_dim, emb_dim=100, hid_dim=100):
        self.input = Input(shape=(None,), name="input")
        self.embedding = Embedding(
            input_dim=input_dim,
            output_dim=emb_dim,
            mask_zero=True,
            name="embedding",
        )
        self.lstm = LSTM(hid_dim, return_sequences=True, name="lstm")
        self.fc = Dense(output_dim, activation="softmax")

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        lstm = self.lstm(embedding)
        y = self.fc(lstm)
        return Model(inputs=x, outputs=y)


class BidirectionalModel:
    """Bi-LSTMを使って、固有表現認識モデルを作成するためのクラス"""

    def __init__(self, input_dim, output_dim, emb_dim=100, hid_dim=100):
        self.input = Input(shape=(None,), name="input")
        self.embedding = Embedding(
            input_dim=input_dim,
            output_dim=emb_dim,
            mask_zero=True,
            name="embedding",
        )
        lstm = LSTM(hid_dim, return_sequences=True, name="lstm")
        self.bilstm = Bidirectional(lstm, name="bilstm")
        self.fc = Dense(output_dim, activation="softmax")

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        lstm = self.bilstm(embedding)
        y = self.fc(lstm)
        return Model(inputs=x, outputs=y)
