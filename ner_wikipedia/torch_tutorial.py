"""Based on
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTMTagger(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ):
        super(LSTMTagger, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: "torch.Tensor") -> "torch.Tensor":
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq: list[str], to_ix: dict[str, int]) -> "torch.Tensor":
    """
    >>> actual = prepare_sequence(["b", "a", "c"], {"a": 0, "b": 1, "c": 2})
    >>> expected = torch.tensor([1, 0, 2], dtype=torch.long)
    >>> torch.equal(actual, expected)
    True
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    (
        "The dog ate the apple".lower().split(),
        ["DET", "NN", "V", "DET", "NN"],
    ),
    ("Everybody read that book".lower().split(), ["NN", "V", "DET", "NN"]),
]

word_to_ix = {}  # type: dict[str, int]
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}


def check_with_sample(sentence: list[str]) -> None:
    inputs = prepare_sequence(sentence, word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
    max_args = torch.argmax(tag_scores, dim=1)
    print("Sentence:", " ".join(sentence))
    print("Predicted tags:", " ".join(ix_to_tag[i] for i in max_args.tolist()))


if __name__ == "__main__":
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    torch.manual_seed(1)

    model = LSTMTagger(
        EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix)
    )
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    with torch.no_grad():
        print("Before training:")
        check_with_sample(training_data[0][0])

    for epoch in range(300):
        for sentence, tags in training_data:
            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            tag_scores = model(sentence_in)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        print("After training:")
        check_with_sample(training_data[0][0])
