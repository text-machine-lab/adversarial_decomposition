from pathlib import Path

DATA_DIR = Path('./data/')
EXPERIMENTS_DIR = DATA_DIR.joinpath('experiments/')

SHAKESPEARE_DATASET_DIR = Path(
    '/home/aromanov/projects/paraphrase_v2/data/datasets/shakespeare/data/align/plays/merged/'
)
YELP_DATASET_DIR = Path(
    '/home/aromanov/projects/paraphrase_v2/data/datasets/language-style-transfer/data/yelp/'
)

WORD_EMBEDDINGS_FILENAMES = dict(
    glove=Path('/home/aromanov/data/word_vectors/glove/glove.840B.300d.pickled'),
    fast_text=Path('/home/aromanov/data/word_vectors/fastText/crawl-300d-2M.pickled'),
)
