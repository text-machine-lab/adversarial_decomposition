from pathlib import Path

DATA_DIR = Path('./data/')
EXPERIMENTS_DIR = DATA_DIR.joinpath('experiments/')

SHAKESPEARE_DATASET_DIR = DATA_DIR.joinpath('datasets/shakespeare/data/align/plays/merged/')
YELP_DATASET_DIR = DATA_DIR.joinpath('datasets/yelp/data/yelp')

WORD_EMBEDDINGS_FILENAMES = dict(
    glove=DATA_DIR.joinpath('word_embeddings/glove.840B.300d.pickled'),
    fast_text=DATA_DIR.joinpath('word_embeddings/crawl-300d-2M.pickled'),
)
