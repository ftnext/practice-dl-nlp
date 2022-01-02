"""Based on
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset


class LSTMTagger(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super(LSTMTagger, self).__init__()

        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.loss_function = nn.NLLLoss()

    def forward(self, sentence: "torch.Tensor") -> "torch.Tensor":
        # Linear layer が batch_size=1 を要求している

        # LightningDataModuleにする前 sentence.size は 4や5
        # LightningDataModuleにした後 sentence.size は [1, 5] （paddingもした）
        embeds = self.word_embeddings(
            sentence.view(-1)
        )  # embeds.size() は [5, 6]
        lstm_out, _ = self.lstm(
            embeds.view(sentence.size(1), sentence.size(0), -1)
        )  # lstm_out.size() は [5, 1, 6]
        tag_space = self.hidden2tag(
            lstm_out.view(sentence.size(1), -1)
        )  # tag_space.size() は [5, 4]
        tag_scores = F.log_softmax(
            tag_space, dim=1
        )  # tag_scores.size() は [5, 4]
        return tag_scores

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        return optimizer

    def training_step(self, train_batch, batch_ids):
        # DataLoaderのbatch_sizeが1（デフォルトのため）batch_idsはint (0, 1)
        sentence_in, targets = train_batch  # sizeはどちらも [1, 5]
        tag_scores = self(sentence_in)  # tag_scores.size() は [5, 4]
        loss = self.loss_function(tag_scores, targets.view(-1))
        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        mean_loss = torch.tensor(
            [o["loss"] for o in training_step_outputs]
        ).mean()
        self.log("loss", mean_loss)


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

word_to_ix = {"<PAD>": 0}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"<PAD>": 0, "DET": 1, "NN": 2, "V": 3}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}


class SentenceTagsPairDataset(IterableDataset):
    def __init__(self):
        super(SentenceTagsPairDataset, self).__init__()

        sentence_tensors, tags_tensors = [], []
        for sentence, tags in training_data:
            sentence_tensors.append(prepare_sequence(sentence, word_to_ix))
            tags_tensors.append(prepare_sequence(tags, tag_to_ix))
        self.sentence_tensors = pad_sequence(
            sentence_tensors, batch_first=True
        )  # size は [2, 5] （pad_sequenceしたのでsize(1)が揃っている）
        self.tags_tensors = pad_sequence(
            tags_tensors, batch_first=True
        )  # size は [2, 5]

    def __iter__(self):
        return zip(self.sentence_tensors, self.tags_tensors)


class PosTaggingExampleDataModule(pl.LightningDataModule):
    def __init__(self, *, batch_size: int) -> None:
        super(PosTaggingExampleDataModule, self).__init__()
        self.batch_size = batch_size

    def setup(self, stage: str | None) -> None:
        self.example_train = SentenceTagsPairDataset()

    def train_dataloader(self):
        # paddingしたので batch_size=2 が指定できるが、モデルのアーキテクチャが非対応
        return DataLoader(self.example_train, batch_size=self.batch_size)


def check_with_sample(sentence: list[str]) -> None:
    inputs = prepare_sequence(sentence, word_to_ix)
    # inputs を PosTaggingExampleDataModule が返す形に変換
    tag_scores = model(inputs.view(-1, len(inputs)))
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

    with torch.no_grad():
        print("Before training:")
        check_with_sample(training_data[0][0])

    callbacks = [EarlyStopping("loss")]
    trainer = pl.Trainer(callbacks=callbacks)
    dm = PosTaggingExampleDataModule(batch_size=1)
    # 以下で、batchごとに optimizer.zero_grad() が呼ばれるので model.zero_grad() は消せる
    trainer.fit(model, datamodule=dm)

    with torch.no_grad():
        print("After training:")
        check_with_sample(training_data[0][0])
