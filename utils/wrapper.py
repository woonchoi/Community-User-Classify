import warnings
warnings.simplefilter("ignore", UserWarning)
from datetime import datetime

import re
import torch
from torch import nn, optim
from torchtext.data import TabularDataset, Iterator, BucketIterator
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from core.model import SeqClassifier
from utils.sequence import init_weights, count_parameters, make_field


def train_model(train_file, batch_size=32, epochs=10,
                emb_dim=512, hid_dim=128, class_num=2,
                bidirectional=None, dropout=None, clip=1,
                save_file=None, vocab_file=None, device='cpu'):
    TEXT, LABEL, _ = make_field(train=True)

    if not train_file.endswith('.csv'):
        raise ValueError(f"'train_file' expected '*.csv', got {train_file}")
    train_data = TabularDataset(
        path=train_file, format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True
    )
    print(f"trainset size: {len(train_data)}")
    #print(vars(train_data[0]))

    TEXT.build_vocab(train_data, min_freq=1)
    torch.save(TEXT.vocab, vocab_file)

    print(f"Unique tokens in text vocabulary: {len(TEXT.vocab)}")
    # print(TEXT.vocab.stoi)

    model = SeqClassifier(input_dim=len(TEXT.vocab), emb_dim=emb_dim,
                          hid_dim=hid_dim, class_num=class_num,
                          bidirectional=bidirectional, dropout=dropout)
    if not isinstance(model, SeqClassifier):
        raise ValueError(f"'model' expected <nn.Module 'SeqClassifier'>, "
                         + f"got {type(model)}")

    print(f"\n[TRAIN] {model.name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model = model.to(device)
    model.apply(init_weights)
    print(model)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    train_set, valid_set = train_data.split([9, 1])

    train_loader, valid_loader = BucketIterator.splits(
        datasets=(train_set, valid_set),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
    )

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0

        with tqdm(train_loader,
                  desc=f"Train Epoch {epoch+1}/{epochs}",
                  ascii=True,
                  colour='green') as tqdm_loader:
            epoch_loss = 0
            model.train()
            for ii, batch in enumerate(tqdm_loader):
                src = batch.text.to(device)
                label = batch.label.to(device)

                output = model(src)

                #label = [batch size]
                #output = [batch size, class num]

                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                epoch_loss += loss.item()

                if (ii+1) == len(train_loader):
                    train_loss = epoch_loss/len(train_loader)
                    tqdm_loader.set_postfix(loss=train_loss)
                else:
                    tqdm_loader.set_postfix(loss=loss.item())

        with tqdm(valid_loader,
                  desc=f"Valid Epoch {epoch+1}/{epochs}",
                  ascii=True,
                  colour='yellow') as tqdm_loader:
            epoch_loss = 0
            with torch.no_grad():
                model.eval()
                for ii, batch in enumerate(tqdm_loader):

                    src = batch.text.to(device)
                    label = batch.label.to(device)

                    output = model(src)

                    #label = [batch size]
                    #output = [batch size, class num]

                    loss = criterion(output, label)

                    epoch_loss += loss.item()

                    if (ii+1) == len(valid_loader):
                        valid_loss = epoch_loss/len(valid_loader)
                        tqdm_loader.set_postfix(loss=valid_loss)
                    else:
                        tqdm_loader.set_postfix(loss=loss.item())

        print(f"train loss: {train_loss:.3f} | valid loss: {valid_loss:.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()

    if save_file is None:
        time = datetime.now()
        save_file = f"model/{time:%Y%m%d_%H%M%S}_weights.pt"
    torch.save(best_model, save_file)


def evaluate_model(test_file, batch_size=32, emb_dim=512, hid_dim=128,
                   class_num=2, bidirectional=None, dropout=None,
                   weight_file=None, vocab_file=None, device='cpu'):
    if not vocab_file.endswith('.pt'):
        raise ValueError(f"'vocab_file' expected '*.pt', got {vocab_file}")
    TEXT, LABEL, _ = make_field()
    TEXT.vocab = torch.load('saved_vocab.pt')

    if not test_file.endswith('.csv'):
        raise ValueError(f"'test_file' expected '*.csv', got {test_file}")
    test_set = TabularDataset(
        path=test_file, format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True
    )
    print(f"testset size: {len(test_set)}")

    model = SeqClassifier(input_dim=len(TEXT.vocab), emb_dim=emb_dim,
                          hid_dim=hid_dim, class_num=class_num,
                          bidirectional=bidirectional, dropout=dropout)
    if not isinstance(model, SeqClassifier):
        raise ValueError(f"'model' expected <nn.Module 'SeqClassifier'>, "
                         + f"got {type(model)}")

    if not weight_file.endswith('.pt'):
        raise ValueError(f"'weight_file' expected '*.pt', got {weight_file}")
    model.load_state_dict(torch.load(weight_file))

    test_loader = Iterator(
        dataset=test_set,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        sort=False,
    )

    predict = []
    label = []

    model = model.to(device)

    with tqdm(test_loader, ascii=True) as tqdm_loader:
        with torch.no_grad():
            model.eval()
            for batch in tqdm_loader:

                src = batch.text.to(device)
                label += batch.label.tolist()

                output = model(src)

                #output = [batch size, class num]

                predict += torch.argmax(output, dim=1).tolist()

    report = classification_report(label, predict,
                                   target_names=['dcinside', 'ruliweb'])
    print(report)


def predict_class(input_text, emb_dim=512, hid_dim=128, class_num=2,
                  bidirectional=None, dropout=None, weight_file=None,
                  vocab_file=None, device='cpu'):
    preprocessed = re.sub(r'[^ 가-힣]', '', input_text)
    if len(preprocessed) <= 5:
        raise ValueError(f"'input text' expected in Korean more than 5 char, "
                         + f"got {input_text}")

    if not vocab_file.endswith('.pt'):
        raise ValueError(f"'vocab_file' expected '*.pt', got {vocab_file}")
    TEXT, _, tokenizer = make_field()
    TEXT.vocab = torch.load('saved_vocab.pt')

    model = SeqClassifier(input_dim=len(TEXT.vocab), emb_dim=emb_dim,
                          hid_dim=hid_dim, class_num=class_num,
                          bidirectional=bidirectional, dropout=dropout)
    if not isinstance(model, SeqClassifier):
        raise ValueError(f"'model' expected <nn.Module 'SeqClassifier'>, "
                         + f"got {type(model)}")

    if not weight_file.endswith('.pt'):
        raise ValueError(f"'weight_file' expected '*.pt', got {weight_file}")
    model.load_state_dict(torch.load(weight_file))

    model = model.to(device)
    model.eval()

    tokenized = tokenizer(preprocessed)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    src = torch.LongTensor(indexed).to(device)
    #src = [src len]
    src = src.unsqueeze(1).T
    #src = [1, src len]
    output = model(src)
    #output = [1, class num]
    softmax = nn.Softmax(dim=1)
    prediction = softmax(output)
    prob = prediction.to('cpu').detach().numpy().squeeze()
    # prob = [dcinside probability, ruliweb probability]

    return prob