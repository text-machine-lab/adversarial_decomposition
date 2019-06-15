from collections import Counter


class Vocab(object):
    END_TOKEN = '<end>'
    START_TOKEN = '<start>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self, special_tokens=None):
        super().__init__()

        self.special_tokens = special_tokens

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        if self.special_tokens is not None:
            self.add_document(self.special_tokens)

    def add_document(self, document, rebuild=True):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        if rebuild:
            self._rebuild_id2token()

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc, rebuild=False)

        self._rebuild_id2token()

    def prune_vocab(self, max_size):
        nb_tokens_before = len(self.token2id)

        tokens_all = set(self.token2id.keys())
        tokens_most_common = set(t for t, c in self.token_counts.most_common(max_size))
        tokens_special = set(self.special_tokens)

        tokens_to_keep = tokens_most_common | tokens_special
        tokens_to_delete = tokens_all - tokens_to_keep

        for token in tokens_to_delete:
            self.token_counts.pop(token)
            # self.token2id.pop(token)

        self.add_document(self.special_tokens, rebuild=False)
        self.add_document(self.token_counts.keys(), rebuild=False)

        self._rebuild_id2token()

        nb_tokens_after = len(self.token2id)

        print(f'Vocab pruned: {nb_tokens_before} -> {nb_tokens_after}')

    def _rebuild_id2token(self):
        self.id2token = {i: t for t, i in self.token2id.items()}

    def get(self, item, default=None):
        return self.token2id.get(item, default)

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return len(self.token2id)

    def __str__(self):
        return f'Vocab: {len(self)} tokens'
