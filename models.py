import numpy as np
import torch
import torch.nn.functional as F

from losses import SequenceReconstructionLoss, StyleEntropyLoss, MeaningZeroLoss
from utils import get_sequences_lengths, to_device


class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers=1, bidirectional=False, return_sequence=False):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def zero_state(self, batch_size):
        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = self.num_layers if not self.bidirectional else self.nb_layers * 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        # shape: (num_layers, batch_size, hidden_dim)
        h = to_device(torch.zeros(*state_shape))

        # shape: (num_layers, batch_size, hidden_dim)
        c = torch.zeros_like(h)

        return h, c

    def forward(self, inputs, lengths):
        batch_size = inputs.shape[0]

        # shape: (num_layers, batch_size, hidden_dim)
        h, c = self.zero_state(batch_size)

        lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
        inputs_sorted = inputs[inputs_sorted_idx]

        # pack sequences
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, lengths_sorted.detach(), batch_first=True)

        # shape: (batch_size, sequence_len, hidden_dim)
        outputs, (h, c) = self.lstm(packed, (h, c))

        # concatenate if bidirectional
        # shape: (batch_size, hidden_dim)
        h = torch.cat([x for x in h], dim=-1)

        # unpack sequences
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
        outputs = outputs[inputs_unsorted_idx]
        h = h[inputs_unsorted_idx]

        if self.return_sequence:
            return outputs
        else:
            return h


class Squeeze(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()

        self.dim = dim

    def forward(self, inputs):
        inputs = inputs.squeeze(self.dim)
        return inputs


class SpaceTransformer(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, output_size),
            torch.nn.Dropout(dropout),
            # torch.nn.ELU(),
            torch.nn.Hardtanh(-10, 10),
        )

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, inputs):
        outputs = self.classifier(inputs)
        return outputs


class Seq2Seq(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, max_len, scheduled_sampling_ratio,
                 start_index, end_index, pad_index, trainable_embeddings, W_emb=None, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.dropout = dropout
        self.scheduled_sampling_ratio = scheduled_sampling_ratio
        self.trainable_embeddings = trainable_embeddings

        self.start_index = start_index
        self.end_index = end_index
        self.pad_index = pad_index

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_index)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(embedding_size, hidden_size, dropout)
        self.decoder_cell = torch.nn.LSTMCell(embedding_size, hidden_size)
        self.output_projection = torch.nn.Linear(hidden_size, vocab_size)

        self._xent_loss = SequenceReconstructionLoss(ignore_index=pad_index)

    def encode(self, inputs):
        # shape: (batch_size, sequence_len)
        sentence = inputs['sentence']

        # shape: (batch_size, )
        lengths = get_sequences_lengths(sentence)

        # shape: (batch_size, sequence_len, embedding_size)
        sentence_emb = self.embedding(sentence)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.encoder(sentence_emb, lengths)

        output_dict = {
            'decoder_hidden': decoder_hidden
        }

        return output_dict

    def decode(self, state, targets=None):
        # shape: (batch_size, hidden_size)
        decoder_hidden = state['decoder_hidden']
        decoder_cell = torch.zeros_like(decoder_hidden)

        batch_size = decoder_hidden.size(0)

        if targets is not None:
            num_decoding_steps = targets.size(1)
        else:
            num_decoding_steps = self.max_len

        # shape: (batch_size, )
        last_predictions = decoder_hidden.new_full((batch_size,), fill_value=self.start_index).long()
        # shape: (batch_size, sequence_len, vocab_size)
        step_logits = []
        # shape: (batch_size, sequence_len, )
        step_predictions = []

        for timestep in range(num_decoding_steps):
            # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio during training.
            # shape: (batch_size,)
            decoder_input = last_predictions
            if timestep > 0 and self.training and torch.rand(1).item() > self.scheduled_sampling_ratio:
                decoder_input = targets[:, timestep - 1]

            # shape: (batch_size, embedding_size)
            decoder_input = self.embedding(decoder_input)

            # shape: (batch_size, hidden_size)
            decoder_hidden, decoder_cell = self.decoder_cell(decoder_input, (decoder_hidden, decoder_cell))

            # shape: (batch_size, vocab_size)
            output_projection = self.output_projection(decoder_hidden)

            # list of tensors, shape: (batch_size, 1, vocab_size)
            step_logits.append(output_projection.unsqueeze(1))

            # shape (predicted_classes): (batch_size,)
            last_predictions = torch.argmax(output_projection, 1)

            # list of tensors, shape: (batch_size, 1)
            step_predictions.append(last_predictions.unsqueeze(1))

        # shape: (batch_size, max_len, vocab_size)
        logits = torch.cat(step_logits, 1)
        # shape: (batch_size, max_len)
        predictions = torch.cat(step_predictions, 1)

        state.update({
            "logits": logits,
            "predictions": predictions,
        })

        return state

    def calc_loss(self, output_dict, inputs):
        # shape: (batch_size, sequence_len)
        targets = inputs['sentence']
        # shape: (batch_size, sequence_len, vocab_size)
        logits = output_dict['logits']

        loss = self._xent_loss(logits, targets)

        output_dict['loss'] = loss

        return output_dict

    def forward(self, inputs):
        state = self.encode(inputs)
        output_dict = self.decode(state, inputs['sentence'])

        output_dict = self.calc_loss(output_dict, inputs)

        return output_dict


class Seq2SeqMeaningStyle(Seq2Seq):
    def __init__(self, meaning_size, style_size, nb_styles, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.meaning_size = meaning_size
        self.style_size = style_size
        self.nb_styles = nb_styles

        self.hidden_meaning = SpaceTransformer(self.hidden_size, self.meaning_size, self.dropout)
        self.hidden_style = SpaceTransformer(self.hidden_size, self.meaning_size, self.dropout)
        self.meaning_style_hidden = SpaceTransformer(meaning_size + style_size, self.hidden_size, self.dropout)

        # D - discriminator: discriminates the style of a sentence
        self.D_meaning = Discriminator(meaning_size, self.hidden_size, nb_styles, self.dropout)
        self.D_style = Discriminator(style_size, self.hidden_size, nb_styles, self.dropout)

        # P - predictor: predicts the meaning of a sentence (word embeddings)
        self.P_meaning = Discriminator(meaning_size, self.hidden_size, self.embedding_size, self.dropout)
        self.P_style = Discriminator(style_size, self.hidden_size, self.embedding_size, self.dropout)

        # P_bow - predictor_bow: predicts the meaning of a sentence (BoW)
        self.P_bow_meaning = Discriminator(meaning_size, self.hidden_size, self.vocab_size, self.dropout)
        self.P_bow_style = Discriminator(style_size, self.hidden_size, self.vocab_size, self.dropout)

        # Discriminator for gaussian z
        self.D_hidden = Discriminator(self.hidden_size, self.hidden_size, 2, self.dropout)

        self._D_loss = torch.nn.CrossEntropyLoss()
        self._D_adv_loss = StyleEntropyLoss()

        self._P_loss = torch.nn.MSELoss()
        self._P_adv_loss = MeaningZeroLoss()

        self._P_bow_loss = torch.nn.BCEWithLogitsLoss()
        self._P_bow_adv_loss = StyleEntropyLoss()

    def encode(self, inputs):
        state = super().encode(inputs)

        # shape: (batch_size, hidden_size)
        decoder_hidden = state['decoder_hidden']

        # shape: (batch_size, hidden_size)
        meaning_hidden = self.hidden_meaning(decoder_hidden)

        # shape: (batch_size, hidden_size)
        style_hidden = self.hidden_style(decoder_hidden)

        state['meaning_hidden'] = meaning_hidden
        state['style_hidden'] = style_hidden

        return state

    def combine_meaning_style(self, state):
        # shape: (batch_size, hidden_size * 2)
        decoder_hidden = torch.cat([state['meaning_hidden'], state['style_hidden']], dim=-1)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.meaning_style_hidden(decoder_hidden)

        state['decoder_hidden'] = decoder_hidden

        return state

    def decode(self, state, targets=None):
        state = self.combine_meaning_style(state)

        output_dict = super().decode(state, targets)
        return output_dict

    def calc_discriminator_loss(self, output_dict, inputs):
        output_dict['loss_D_meaning'] = self._D_loss(output_dict['D_meaning_logits'], inputs['style'])
        output_dict['loss_D_style'] = self._D_loss(output_dict['D_style_logits'], inputs['style'])

        if 'meaning_embedding' in inputs:
            output_dict['loss_P_meaning'] = self._P_loss(output_dict['P_meaning'], inputs['meaning_embedding'])
            output_dict['loss_P_style'] = self._P_loss(output_dict['P_style'], inputs['meaning_embedding'])

        if 'meaning_bow' in inputs:
            output_dict['loss_P_bow_meaning'] = self._P_bow_loss(output_dict['P_bow_meaning'], inputs['meaning_bow'])
            output_dict['loss_P_bow_style'] = self._P_bow_loss(output_dict['P_bow_style'], inputs['meaning_bow'])

        return output_dict

    def calc_discriminator_adv_loss(self, output_dict, inputs):
        output_dict['loss_D_adv_meaning'] = self._D_adv_loss(output_dict['D_meaning_logits'])
        output_dict['loss_D_adv_style'] = self._D_loss(output_dict['D_style_logits'], inputs['style'])

        if 'meaning_embedding' in inputs:
            output_dict['loss_P_adv_meaning'] = self._P_loss(output_dict['P_meaning'], inputs['meaning_embedding'])
            output_dict['loss_P_adv_style'] = self._P_adv_loss(output_dict['P_style'])

        if 'meaning_bow' in inputs:
            output_dict['loss_P_bow_adv_meaning'] = self._P_bow_loss(
                output_dict['P_bow_meaning'], inputs['meaning_bow'])
            output_dict['loss_P_bow_adv_style'] = self._P_bow_adv_loss(output_dict['P_bow_style'])

        return output_dict

    def discriminate(self, output_dict, inputs, adversarial=False):
        output_dict['D_meaning_logits'] = self.D_meaning(output_dict['meaning_hidden'])
        output_dict['D_style_logits'] = self.D_style(output_dict['style_hidden'])

        if 'meaning_embedding' in inputs:
            output_dict['P_meaning'] = self.P_meaning(output_dict['meaning_hidden'])
            output_dict['P_style'] = self.P_style(output_dict['style_hidden'])

        if 'meaning_bow' in inputs:
            output_dict['P_bow_meaning'] = self.P_bow_meaning(output_dict['meaning_hidden'])
            output_dict['P_bow_style'] = self.P_bow_style(output_dict['style_hidden'])

        # calc loss
        if not adversarial:
            output_dict = self.calc_discriminator_loss(output_dict, inputs)
        else:
            output_dict = self.calc_discriminator_adv_loss(output_dict, inputs)

        return output_dict


class StyleClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, trainable_embeddings, pad_index, nb_styles,
                 W_emb=None, **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.trainable_embeddings = trainable_embeddings
        self.nb_styles = nb_styles

        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_index)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(embedding_size, hidden_size, dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_size, nb_styles),
        )

        self._xent_loss = torch.nn.CrossEntropyLoss()

    def encode(self, inputs):
        # shape: (batch_size, sequence_len)
        sentence = inputs['sentence']

        # shape: (batch_size, )
        lengths = get_sequences_lengths(sentence)

        # shape: (batch_size, sequence_len, embedding_size)
        sentence_emb = self.embedding(sentence)

        # shape: (batch_size, hidden_size)
        decoder_hidden = self.encoder(sentence_emb, lengths)

        output_dict = {
            'decoder_hidden': decoder_hidden
        }

        return output_dict

    def classify(self, state):
        # shape: (batch_size, hidden_size)
        hidden = state['decoder_hidden']

        # shape: (batch_size, nb_classes)
        logits = self.classifier(hidden)
        predictions = torch.argmax(logits, 1)

        state.update({
            "logits": logits,
            "predictions": predictions,
        })

        return state

    def calc_loss(self, output_dict, inputs):
        # shape: (batch_size, sequence_len)
        targets = inputs['style']
        # shape: (batch_size, sequence_len, vocab_size)
        logits = output_dict['logits']

        loss = self._xent_loss(logits, targets)

        output_dict['loss'] = loss

        return output_dict

    def forward(self, inputs):
        state = self.encode(inputs)
        output_dict = self.classify(state)

        output_dict = self.calc_loss(output_dict, inputs)

        return output_dict
