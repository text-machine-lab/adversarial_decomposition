# Adversarial Decomposition of Text Representation
The code for the paper "Adversarial Decomposition of Text Representation", NAACL 2019 
https://arxiv.org/abs/1808.09042

# Installation 

 1. Clone this repo: `https://github.com/text-machine-lab/adversarial_decomposition.git`
 2. Install NumPy: `pip install numpy==1.16.3`
 3. Install PyTorch v1.1.0: `pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl` (for python3.6, use `pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl`)
 4. Install dependencies: `pip install -r requirements.txt`
 5. Download spacy models: `python -m spacy download en_core_web_lg`

# Initial setup

 1. Create dir `mkdir -p data/experiments`
 2. Create dir `mkdir -p data/datasets`
 3. Create dir `mkdir -p data/word_embeddings`
 3. Download the Shakespeare data: `git clone https://github.com/cocoxu/Shakespeare.git data/datasets/shakespeare`
 3. Download the Yelp data: `git clone https://github.com/shentianxiao/language-style-transfer.git data/datasets/yelp`
 4. Download the pickled GloVe embeddings `wget https://mednli.blob.core.windows.net/shared/word_embeddings/glove.840B.300d.pickled -O data/word_embeddings/glove.840B.300d.pickled`
4. Download the pickled fastText embeddings `wget https://mednli.blob.core.windows.net/shared/word_embeddings/crawl-300d-2M.pickled -O data/word_embeddings/crawl-300d-2M.pickled`

# Running the code

Global constants are set in the file `settings.py`. In general, you don't need to change this file.
Experiment parameters are set in the `config.py` file. 

First, run the preprocessing script: `python preprocess.py`
This scipt will print the ID of the preprocessing experiment, for example `preprocess.buppgpnf`. Copy this ID and change parameter `preprocess_exp_id` of the `TrainConfig` class on the line 12 in the file `config.py` file accordingly.

After you set the preprocess experiment id, run the training: `python train.py`.
This scirpt will also print the ID of the training experiment. You can paste it in the `eval_generation.ipynb` notebook to play with the model.

## Chaning the form and meaning
The provided `eval_generation.ipynb` notebook shows how to use the model to swap the meaning and form vectors of the input sentences!


# Citation
If you find this code helpful, please consider citing our paper:

*A. Romanov, A. Rumshisky, A. Rogers, D. Donahue,Adversarial decomposition of text represen-tation, In Proceedings of NAACL 2019: Conference of the North American Chapter of the Association for Computational Linguistics, 2019*

https://arxiv.org/abs/1808.09042

```
@inproceedings{romanov2019adversarial,
  title={Adversarial Decomposition of Text Representation},
  author={Romanov, Alexey and Rumshisky, Anna and Rogers, Anna and Donahue, David},
  booktitle={Proceedings of NAACL 2019: Conference of the North American Chapter of the Association for Computational Linguistics},
  year={2019}
}
```

