import abc
import csv
import dataclasses
import itertools
import re

import spacy
import numpy as np
import torch.utils.data

from vocab import Vocab


class SentenceStyleDatasetReader(object):

    def __init__(self, min_len, max_len, lowercase, *args, **kwargs):
        self.min_len = min_len
        self.max_len = max_len
        self.lowercase = lowercase

        disable = ['vectors', 'textcat', 'tagger', 'parser', 'ner']
        self.spacy = spacy.load('en_core_web_lg', disable=disable)
        self.spacy.add_pipe(self.spacy.create_pipe('sentencizer'))

    @abc.abstractmethod
    def _read(self, data_path):
        pass


    def clean_sentence(self, sentence):
        sentence_cleaned = sentence.replace('\r', ' ')
        sentence_cleaned = sentence_cleaned.replace('\n', ' ')
        sentence_cleaned = sentence_cleaned.replace("'m", ' am')
        sentence_cleaned = sentence_cleaned.replace("'ve", ' have')
        sentence_cleaned = sentence_cleaned.replace("n\'t", ' not')
        sentence_cleaned = sentence_cleaned.replace("\'re", ' are')
        sentence_cleaned = sentence_cleaned.replace("\'d", ' would')
        sentence_cleaned = sentence_cleaned.replace("\'ll", ' will')

        return sentence_cleaned

    def preprocess_sentence(self, sentence):
        sentence = [
            token.lower_ if self.lowercase else token.text
            for token in sentence
            if not token.is_space
        ]

        # cut to max len -1 for the END token
        sentence = sentence[:self.max_len - 1]

        return sentence

    def read(self, data_path):
        samples = []

        for sentence, style in self._read(data_path):
            sentence = self.clean_sentence(sentence)
            sentence = self.spacy(sentence)
            sentence = self.preprocess_sentence(sentence)

            if len(sentence) > self.min_len:
                sample = dict(sentence=sentence, style=style)
                samples.append(sample)

        return samples


class ShakespeareDatasetReader(SentenceStyleDatasetReader):

    def _read(self, data_path):
        for file in data_path.iterdir():
            file_style = '-'
            if file.name.endswith('original.snt.aligned'):
                file_style = 'original'
            if file.name.endswith('modern.snt.aligned'):
                file_style = 'modern'

            with open(file) as f:
                for line in f:
                    sentence = line.strip()

                    yield sentence, file_style


class YelpDatasetReader(SentenceStyleDatasetReader):
    def clean_sentence(self, sentence):
        sentence = super().clean_sentence(sentence)

        sentence_cleaned = sentence.replace("_num_", 'number')

        return sentence_cleaned

    def _read(self, data_path):
        files = [
            data_path.joinpath('sentiment.train.0'),
            data_path.joinpath('sentiment.train.1'),
            data_path.joinpath('sentiment.dev.0'),
            data_path.joinpath('sentiment.dev.1'),
        ]

        for file in files:
            file_style = '-'
            if file.name.endswith('0'):
                file_style = 'negative'
            if file.name.endswith('1'):
                file_style = 'positive'

            with open(file) as f:
                for line in f:
                    sentence = line.strip()

                    yield sentence, file_style



class SentenceStyleDataset(torch.utils.data.Dataset):

    def __init__(self, instances, vocab, style_vocab):
        self.instances = instances
        self.vocab = vocab
        self.style_vocab = style_vocab

        self.max_len = max(len(inst['sentence']) for inst in instances) + 1  # +1 for the END token

        for inst in self.instances:
            inst_encoded = self.encode_instance(inst)
            inst.update(inst_encoded)

    def pad_sentence(self, sentence):
        # add end token
        sentence = sentence + [Vocab.END_TOKEN, ]

        # pad
        sentence = sentence + [Vocab.PAD_TOKEN, ] * (self.max_len - len(sentence))

        return sentence

    def encode_instance(self, instance):
        sentence, style = instance['sentence'], instance['style']

        sentence = self.pad_sentence(sentence)
        sentence_enc = np.array([self.vocab.get(t, Vocab.UNK_TOKEN) for t in sentence], dtype=np.long)

        style_enc = self.style_vocab[style]

        encoded = dict(
            sentence_enc=sentence_enc,
            style_enc=style_enc
        )
        return encoded

    def __getitem__(self, index):
        inst = self.instances[index]

        inst = {
            'sentence': inst['sentence_enc'],
            'style': inst['style_enc'],
        }

        return inst

    def __len__(self):
        return len(self.instances)


class MeaningEmbeddingSentenceStyleDataset(SentenceStyleDataset):
    def __init__(self, W_emb, style_dimensions, style_tokens_proportion, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.W_emb = W_emb
        self.style_dimensions = style_dimensions
        self.style_tokens_proportion = style_tokens_proportion

        for inst in self.instances:
            inst['meaning_embedding'] = self.calc_meaning_embedding(inst, W_emb)

    def calc_meaning_embedding(self, instance, W_emb):
        tokens = [t for t in instance['sentence'] if t not in {Vocab.END_TOKEN, Vocab.PAD_TOKEN, Vocab.UNK_TOKEN}]

        nb_tokens = len(tokens)
        nb_style_tokens = int(np.ceil(nb_tokens * self.style_tokens_proportion))

        sentence_embedding = np.array([W_emb[self.vocab[t]] for t in tokens])
        sorted_by_style_dim_idx = np.argsort(-np.abs(sentence_embedding[:, self.style_dimensions]).max(axis=-1))
        meaning_idx = sorted_by_style_dim_idx[nb_style_tokens:]
        meaning_embedding = np.sum(sentence_embedding[meaning_idx], axis=0) / (nb_tokens - nb_style_tokens)

        return meaning_embedding

    def __getitem__(self, index):
        inst = super().__getitem__(index)

        inst['meaning_embedding'] = self.instances[index]['meaning_embedding']

        return inst

