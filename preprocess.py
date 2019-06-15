import dataclasses

from sklearn.model_selection import train_test_split

from config import PreprocessConfig
from datasets import MeaningEmbeddingSentenceStyleDataset
from experiment import Experiment
from settings import EXPERIMENTS_DIR
from utils import save_pickle, load_pickle, load_embeddings, create_embeddings_matrix, extract_word_embeddings_style_dimensions
from vocab import Vocab


def save_dataset(exp, dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb):
    save_pickle((dataset_train, dataset_val, dataset_test), exp.experiment_dir.joinpath('datasets.pkl'))
    save_pickle((vocab, style_vocab), exp.experiment_dir.joinpath('vocabs.pkl'))
    save_pickle(W_emb, exp.experiment_dir.joinpath('W_emb.pkl'))

    print(f'Saved: {exp.experiment_dir}')


def load_dataset(exp):
    dataset_train, dataset_val, dataset_test = load_pickle(exp.experiment_dir.joinpath('datasets.pkl'))
    vocab, style_vocab = load_pickle(exp.experiment_dir.joinpath('vocabs.pkl'))
    W_emb = load_pickle(exp.experiment_dir.joinpath('W_emb.pkl'))

    print(f'Dataset: {len(dataset_train)}, val: {len(dataset_val)}, test: {len(dataset_test)}')
    print(f'Vocab: {len(vocab)}, style vocab: {len(style_vocab)}')
    print(f'W_emb: {W_emb.shape}')

    return dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb


def create_dataset_reader(cfg):
    dataset_reader_class = cfg.dataset_reader_class

    dataset_reader_params = dataclasses.asdict(cfg)
    dataset_reader = dataset_reader_class(**dataset_reader_params)

    return dataset_reader


def create_vocab(instances):
    vocab = Vocab([Vocab.PAD_TOKEN, Vocab.START_TOKEN, Vocab.END_TOKEN, Vocab.UNK_TOKEN, ])
    vocab.add_documents([inst['sentence'] for inst in instances])

    style_vocab = Vocab()
    style_vocab.add_document([inst['style'] for inst in instances])

    return vocab, style_vocab


def create_splits(cfg, instances):
    if cfg.test_size != 0:
        instances_train_val, instances_test = train_test_split(instances, test_size=cfg.test_size, random_state=42)
    else:
        instances_test = []
        instances_train_val = instances

    if cfg.val_size != 0:
        instances_train, instances_val = train_test_split(instances_train_val, test_size=cfg.val_size, random_state=0)
    else:
        instances_train = []
        instances_val = []

    return instances_train, instances_val, instances_test


def main(cfg):
    with Experiment(EXPERIMENTS_DIR, cfg, prefix='preprocess') as exp:
        print(f'Experiment started: {exp.experiment_id}')

        # read instances
        dataset_reader = create_dataset_reader(exp.config)
        print(f'Dataset reader: {dataset_reader.__class__.__name__}')

        instances = dataset_reader.read(exp.config.data_path)
        print(f'Instances: {len(instances)}')

        # create vocabularies
        vocab, style_vocab = create_vocab(instances)
        print(f'Vocab: {len(vocab)}, style vocab: {style_vocab}')

        if exp.config.max_vocab_size != 0:
            vocab.prune_vocab(exp.config.max_vocab_size)

        # create splits
        instances_train, instances_val, instances_test = create_splits(exp.config, instances)
        print(f'Train: {len(instances_train)}, val: {len(instances_val)}, test: {len(instances_test)}')

        # create embeddings
        word_embeddings = load_embeddings(cfg)
        W_emb = create_embeddings_matrix(word_embeddings, vocab)

        # extract style dimensions
        style_dimensions = extract_word_embeddings_style_dimensions(cfg, instances_train, vocab, style_vocab, W_emb)

        # create datasets
        dataset_train = MeaningEmbeddingSentenceStyleDataset(
            W_emb, style_dimensions, exp.config.style_tokens_proportion,
            instances_train, vocab, style_vocab
        )
        dataset_val = MeaningEmbeddingSentenceStyleDataset(
            W_emb, style_dimensions, exp.config.style_tokens_proportion,
            instances_val, vocab, style_vocab
        )
        dataset_test = MeaningEmbeddingSentenceStyleDataset(
            W_emb, style_dimensions, exp.config.style_tokens_proportion,
            instances_test, vocab, style_vocab
        )

        save_dataset(exp, dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb)

        print(f'Experiment finished: {exp.experiment_id}')


if __name__ == '__main__':
    main(PreprocessConfig())
