import tempfile
from pathlib import Path

from utils import save_pickle, load_pickle


class Experiment(object):
    _CONFIG_FILENAME = 'config.pkl'

    def __init__(self, experiments_dir, config, prefix=None):
        self.config = config
        self.experiments_dir = experiments_dir
        self.prefix = prefix

        # create dir for the experiment
        if self.prefix is not None:
            self.prefix = f'{self.prefix}.'

        self.experiment_dir = None
        self.experiment_id = None

    def __enter__(self):
        self.experiment_dir = Path(tempfile.mkdtemp(dir=self.experiments_dir, prefix=self.prefix))
        self.experiment_id = self.experiment_dir.name

        # save the config file
        Experiment._save_config(self.config, self.experiment_dir)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @classmethod
    def load(cls, experiments_dir, experiment_id):
        experiment_dir = experiments_dir.joinpath(experiment_id)

        config = Experiment._load_config(experiment_dir)

        exp = Experiment(experiments_dir, config)
        exp.experiment_dir = experiment_dir
        exp.experiment_id = exp.experiment_dir.name

        return exp

    @classmethod
    def _save_config(cls, config, experiment_dir):
        filename = experiment_dir.joinpath(Experiment._CONFIG_FILENAME)
        save_pickle(config, filename)

    @classmethod
    def _load_config(cls, experiment_dir):
        filename = experiment_dir.joinpath(Experiment._CONFIG_FILENAME)
        config = load_pickle(filename)

        return config
