import pandas as pd
from torch import nn
from torchtext.data import Field
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_tokenizer
import matplotlib.pyplot as plt


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_field(train=False):
    if train:
        generate_sp_model('data/trainset.csv', vocab_size=5000, model_type='bpe', model_prefix='sentpi')
    sp_model = load_sp_model('sentpi.model')
    sp_tokens_generator = sentencepiece_tokenizer(sp_model)
    tokenizer = lambda x: list(sp_tokens_generator([x]))[0]

    TEXT = Field(sequential=True,
                 use_vocab=True,
                 tokenize=tokenizer,
                 batch_first=True)

    LABEL = Field(sequential=False,
                  use_vocab=False,
                  is_target=True)

    return TEXT, LABEL, tokenizer


def view_result(prob, chart_mode):
    df = pd.DataFrame(
        {
            'prob': prob,
            'label': ['dcinside', 'ruliweb'],
        }
    )
    df = df.sort_values(by='prob', ascending=False, ignore_index=True)
    colors = ['#6699CC', 'silver']
    explode = [0.1, 0.1]
    if chart_mode == 'bar':
        plt.bar(df.index, df.prob)
        plt.title('Probability of input text')
        plt.xlabel('Community')
        plt.ylabel('Probability')
        plt.xticks(df.index, labels=df.label)
        for ii, pp in enumerate(df.prob):
            plt.text(ii, pp+0.01, f'{pp*100:.3f}%', ha='center')
        plt.show()
    elif chart_mode == 'pie':
        wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 1}
        plt.pie(df.prob, labels=df.label, autopct='%.3f%%',
                startangle=90, counterclock=False,
                colors=colors, wedgeprops=wedgeprops)
        plt.show()