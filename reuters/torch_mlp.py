import numpy as np
from tensorflow.keras.datasets import reuters
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import random_split

from keras_mlp import TokenizePreprocessor


np.random.seed(42)
torch.manual_seed(1234)

MAX_WORDS = 1000
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 5


def convert_to_torch_tensors(texts, labels):
    torch_tensors = []
    for text, label in zip(texts, labels):
        text_tensor = torch.tensor(text)
        torch_tensor = (label, text_tensor)
        torch_tensors.append(torch_tensor)
    return torch_tensors


class MLPNet(nn.Module):
    def __init__(self, max_words, number_of_classes, drop_out):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(max_words, 512)
        self.fc2 = nn.Linear(512, number_of_classes)
        self.dropout1 = nn.Dropout(drop_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        return self.fc2(x)  # torch.nn.CrossEntropyLossにsoftmaxの計算が含まれる


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=MAX_WORDS)

tokenizer = TokenizePreprocessor.initialize_tokenizer(MAX_WORDS)
preprocessor = TokenizePreprocessor(tokenizer)
x_train = preprocessor.convert_text_to_matrix(x_train, "binary")
x_test = preprocessor.convert_text_to_matrix(x_test, "binary")

number_of_classes = np.max(y_train) + 1

train_dataset = convert_to_torch_tensors(x_train, y_train)
test_dataset = convert_to_torch_tensors(x_test, y_test)

train_length = int(len(train_dataset) * 0.9)
train_dataset, val_dataset = random_split(
    train_dataset, [train_length, len(train_dataset) - train_length]
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)

device = "gpu" if torch.cuda.is_available() else "cpu"
net = MLPNet(MAX_WORDS, number_of_classes, DROPOUT).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters()
)  # tf.optimizer.Adamのデフォルトの学習率はkerasと同じ

train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

for epoch in range(EPOCHS):
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

    net.train()
    for i, (labels, text_tensors) in enumerate(train_loader):
        labels, text_tensors = labels.to(device), text_tensors.to(device)
        optimizer.zero_grad()
        # mode="binary"で指定したことでdouble(torch.float64)が渡ってきてエラーになることへの対応
        outputs = net(text_tensors.float())
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        acc = (outputs.max(1)[1] == labels).sum()
        train_acc += acc.item()
        loss.backward()
        optimizer.step()
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

    net.eval()
    with torch.no_grad():
        for labels, texts in val_loader:
            labels, texts = labels.to(device), texts.to(device)
            outputs = net(texts.float())
            loss = criterion(outputs, labels)
            val_loss += loss.sum()
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_acc = val_acc / len(val_loader.dataset)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], ",
        f"Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, ",
        f"val Loss: {avg_val_loss:.4f}, val Acc: {avg_val_acc:.4f}",
    )

    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

net.eval()
with torch.no_grad():
    total = 0
    test_acc = 0
    for labels, texts in test_loader:
        labels, texts = labels.to(device), texts.to(device)
        outputs = net(texts.float())
        test_acc += (outputs.max(1)[1] == labels).sum().item()
        total += labels.size(0)
    print(f"test_accuracy: {100 * test_acc / total} %")
