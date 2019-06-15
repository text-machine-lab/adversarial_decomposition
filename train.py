import dataclasses
import itertools
from collections import defaultdict

import numpy as np
import torch.utils.data
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from tensorboardX import SummaryWriter

from config import TrainConfig
from experiment import Experiment
from models import Seq2Seq, Seq2SeqMeaningStyle, StyleClassifier
from preprocess import load_dataset
from settings import EXPERIMENTS_DIR
from update_functions import Seq2SeqUpdateFunction, StyleClassifierUpdateFunction, Seq2SeqMeaningStyleUpdateFunction
from utils import to_device, save_weights, init_weights
from vocab import Vocab



def create_model(cfg, vocab, style_vocab, max_len, W_emb=None):
    model_class = cfg.model_class
    model_params = dataclasses.asdict(cfg)
    model_params.update(dict(
        max_len=max_len,
        vocab_size=len(vocab),
        start_index=vocab[Vocab.START_TOKEN],
        end_index=vocab[Vocab.END_TOKEN],
        pad_index=vocab[Vocab.PAD_TOKEN],
        nb_styles=len(style_vocab),
    ))

    if cfg.pretrained_embeddings:
        model_params.update(dict(
            W_emb=W_emb,
        ))

    model = model_class(**model_params)

    init_weights(model)

    model = to_device(model)

    return model


def create_update_function(cfg, model):
    update_function_class = None
    if isinstance(model, Seq2Seq):
        update_function_class = Seq2SeqUpdateFunction
    if isinstance(model, Seq2SeqMeaningStyle):
        update_function_class = Seq2SeqMeaningStyleUpdateFunction
    if isinstance(model, StyleClassifier):
        update_function_class = StyleClassifierUpdateFunction

    update_function_params = dataclasses.asdict(cfg)
    update_function_params.update(dict(
        model=model,
    ))

    update_function_train = update_function_class(train=True, **update_function_params)
    update_function_eval = update_function_class(train=False, **update_function_params)

    return update_function_train, update_function_eval


class LossAggregatorMetric(Metric):
    def __init__(self, *args, **kwargs):
        self.total_losses = defaultdict(float)
        self.num_updates = defaultdict(int)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.total_losses = defaultdict(float)
        self.num_updates = defaultdict(int)

    def update(self, output):
        for name, val in output.items():
            self.total_losses[name] += float(val)
            self.num_updates[name] += 1

    def compute(self):
        losses = {name: val / self.num_updates[name] for name, val in self.total_losses.items()}

        return losses


def log_progress(epoch, iteration, losses, mode='train', tensorboard_writer=None, use_iteration=False):
    if not use_iteration:
        losses_str = [
            f'{name}: {val:.3f}'
            for name, val in losses.items()
        ]
        losses_str = ' | '.join(losses_str)

        epoch_str = f'Epoch [{epoch}|{iteration}] {mode}'

        print(f'{epoch_str:<25}{losses_str}')

    for name, val in losses.items():
        tensorboard_writer.add_scalar(f'{mode}/{name}', val, epoch if not use_iteration else iteration)


def main(cfg):
    with Experiment(EXPERIMENTS_DIR, cfg, prefix='train') as exp:
        print(f'Experiment started: {exp.experiment_id}')

        preprocess_exp = Experiment.load(EXPERIMENTS_DIR, exp.config.preprocess_exp_id)
        dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb = load_dataset(preprocess_exp)

        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=exp.config.batch_size, shuffle=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=exp.config.batch_size, shuffle=False)
        print(f'Data loader: {len(data_loader_train)}, {len(data_loader_val)}')

        model = create_model(exp.config, vocab, style_vocab, dataset_train.max_len, W_emb)

        update_function_train, update_function_eval = create_update_function(exp.config, model)

        trainer = Engine(update_function_train)
        evaluator = Engine(update_function_eval)

        metrics = {'loss': LossAggregatorMetric(), }
        for metric_name, metric in metrics.items():
            metric.attach(evaluator, metric_name)

        best_loss = np.inf

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_iter(engine):
            losses_train = engine.state.output
            log_progress(trainer.state.epoch, trainer.state.iteration, losses_train, 'train', tensorboard_writer, True)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            nonlocal best_loss

            # evaluator.run(data_loader_train)
            # losses_train = evaluator.state.metrics['loss']

            evaluator.run(data_loader_val)
            losses_val = evaluator.state.metrics['loss']

            # log_progress(trainer.state.epoch, trainer.state.iteration, losses_train, 'train', tensorboard_writer)
            log_progress(trainer.state.epoch, trainer.state.iteration, losses_val, 'val', tensorboard_writer)

            if losses_val[exp.config.best_loss] < best_loss:
                best_loss = losses_val[exp.config.best_loss]
                save_weights(model, exp.experiment_dir.joinpath('best.th'))

        tensorboard_dir = exp.experiment_dir.joinpath('log')
        tensorboard_writer = SummaryWriter(str(tensorboard_dir))

        trainer.run(data_loader_train, max_epochs=exp.config.num_epochs)

        print(f'Experiment finished: {exp.experiment_id}')


if __name__ == '__main__':
    main(TrainConfig())
