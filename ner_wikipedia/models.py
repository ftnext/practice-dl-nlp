from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM


class UnidirectionalModel:
    """LSTMを使ったモデルを作成するためのクラス"""

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
