import dataclasses
from pathlib import Path

from datasets import ShakespeareDatasetReader, YelpDatasetReader
from models import Seq2Seq, Seq2SeqMeaningStyle, StyleClassifier
from settings import SHAKESPEARE_DATASET_DIR, YELP_DATASET_DIR


@dataclasses.dataclass
class TrainConfig:
    model_class: type = Seq2SeqMeaningStyle
    preprocess_exp_id: str = 'preprocess.buppgpnf'  # Shakespeare: xxx | Yelp: 2p089c54

    embedding_size: int = 300
    hidden_size: int = 256
    dropout: float = 0.2
    scheduled_sampling_ratio: float = 0.5
    pretrained_embeddings: bool = True
    trainable_embeddings: bool = False

    meaning_size: int = 128
    style_size: int = 128

    lr: float = 0.001
    weight_decay: float = 0.0000001
    grad_clipping: float = 5

    D_num_iterations: int = 10
    D_loss_multiplier: float = 1
    P_loss_multiplier: float = 10
    P_bow_loss_multiplier: float = 1
    use_discriminator: bool = True
    use_predictor: bool = False
    use_predictor_bow: bool = True
    use_motivator: bool = True
    use_gauss: bool = False

    num_epochs: int = 500
    batch_size: int = 1024
    best_loss: str = 'loss'


@dataclasses.dataclass
class PreprocessConfig:
    data_path: Path = YELP_DATASET_DIR
    dataset_reader_class: type = YelpDatasetReader

    min_len: int = 3
    max_len: int = 20
    lowercase: bool = True
    word_embeddings: str = 'fast_text'
    max_vocab_size: int = 50000

    nb_style_dims: int = 50
    nb_style_dims_sentences: int = 50000
    style_tokens_proportion: float = 0.2

    test_size: int = 10000
    val_size: int = 10000
