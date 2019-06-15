import abc
import itertools

import torch

from utils import to_device


class UpdateFunction(object):
    def __init__(self, model, lr=0.001, weight_decay=0.0000001, grad_clipping=0, training=True, **kwargs):
        self.model = model
        self.grad_clipping = grad_clipping
        self.lr = lr
        self.weight_decay = weight_decay
        self.training = training

    def get_parameters(self, *modules):
        parameters = itertools.chain.from_iterable(m.parameters() for m in modules)
        return parameters

    @abc.abstractmethod
    def step(self, engine, batch):
        pass

    def __call__(self, engine, batch):
        if self.training:
            self.model.train()
        else:
            self.model.eval()

        batch = to_device(batch)

        losses = self.step(engine, batch)

        return losses


class Seq2SeqUpdateFunction(UpdateFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = self.get_parameters(self.model)
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def step(self, engine, batch):
        self.optimizer.zero_grad()

        output_dict = self.model(batch)

        loss = output_dict["loss"]

        if self.training:
            loss.backward()

            if self.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clipping)

            self.optimizer.step()

        output_dict = {
            'loss': loss,
        }
        return output_dict


class Seq2SeqMeaningStyleUpdateFunction(UpdateFunction):
    def __init__(self, D_num_iterations, D_loss_multiplier, P_loss_multiplier, P_bow_loss_multiplier,
                 use_discriminator, use_predictor, use_predictor_bow, use_motivator, use_gauss,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.D_num_iterations = D_num_iterations
        self.D_loss_multiplier = D_loss_multiplier
        self.P_loss_multiplier = P_loss_multiplier
        self.P_bow_loss_multiplier = P_bow_loss_multiplier
        self.use_discriminator = use_discriminator
        self.use_predictor = use_predictor
        self.use_predictor_bow = use_predictor_bow
        self.use_motivator = use_motivator
        self.use_gauss = use_gauss

        self.params_encoder_decoder = self.get_parameters(
            # encoder
            self.model.embedding, self.model.encoder,
            self.model.hidden_meaning, self.model.hidden_style,

            # decoder
            self.model.decoder_cell, self.model.output_projection,
            self.model.meaning_style_hidden
        )

        params_D = []
        if use_discriminator:
            params_D.append(self.model.D_meaning)
            if use_motivator:
                params_D.append(self.model.D_style)

        if use_predictor:
            params_D.append(self.model.P_style)
            if use_motivator:
                params_D.append(self.model.P_meaning)

        if use_predictor_bow:
            params_D.append(self.model.P_bow_style)
            if use_motivator:
                params_D.append(self.model.P_bow_meaning)

        if use_gauss:
            params_D.append(self.model.D_hidden)

        self.params_D = self.get_parameters(*params_D)

        self.optimizer_encoder_decoder = torch.optim.Adam(
            self.params_encoder_decoder, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True
        )
        self.optimizer_D = torch.optim.Adam(
            self.params_D, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True
        )

    def step(self, engine, batch):
        losses_output_dict = {}

        # discriminator
        state = self.model.encode(batch)
        output_dict_D = self.model.discriminate(state, batch)

        if self.use_discriminator:
            loss_D_meaning = output_dict_D['loss_D_meaning']
            losses_output_dict['loss_D_meaning'] = float(loss_D_meaning.item())

            if self.use_motivator:
                loss_D_style = output_dict_D['loss_D_style']
                losses_output_dict['loss_D_style'] = float(loss_D_style.item())
            else:
                loss_D_style = 0
        else:
            loss_D_meaning = 0
            loss_D_style = 0

        if 'meaning_embedding' in batch and self.use_predictor:
            loss_P_style = output_dict_D['loss_P_style']
            losses_output_dict['loss_P_style'] = float(loss_P_style.item())

            if self.use_motivator:
                loss_P_meaning = output_dict_D['loss_P_meaning']
                losses_output_dict['loss_P_meaning'] = float(loss_P_meaning.item())
            else:
                loss_P_meaning = 0
        else:
            loss_P_style = 0
            loss_P_meaning = 0

        if 'meaning_bow' in batch and self.use_predictor_bow:
            loss_P_bow_style = output_dict_D['loss_P_bow_style']
            losses_output_dict['loss_P_bow_style'] = float(loss_P_bow_style.item())

            if self.use_motivator:
                loss_P_bow_meaning = output_dict_D['loss_P_bow_meaning']
                losses_output_dict['loss_P_bow_meaning'] = float(loss_P_bow_meaning.item())
            else:
                loss_P_bow_meaning = 0
        else:
            loss_P_bow_style = 0
            loss_P_bow_meaning = 0

        if self.use_gauss:
            output_dict_D = self.model.combine_meaning_style(output_dict_D)
            G_real = torch.randn_like(output_dict_D['decoder_hidden'])
            G_fake = output_dict_D['decoder_hidden']
            G_labels = torch.cat([torch.ones_like(batch['style']), torch.zeros_like(batch['style'])], dim=0)
            G_inputs = torch.cat([G_real, G_fake], dim=0)

            G_logits = self.model.D_hidden(G_inputs)
            loss_D_hidden = self.model._D_loss(G_logits, G_labels)
            losses_output_dict['loss_D_hidden'] = float(loss_D_hidden.item())
        else:
            loss_D_hidden = 0

        loss_D_total = loss_D_meaning + loss_D_style \
                       + loss_P_meaning + loss_P_style \
                       + loss_P_bow_meaning + loss_P_bow_style \
                       + loss_D_hidden

        if self.training:
            loss_D_total.backward()

            if self.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(self.params_D, self.grad_clipping)

            self.optimizer_D.step()
            self.model.zero_grad()

        # encoder-decoder
        if not self.training or engine.state.iteration % self.D_num_iterations == 0:
            output_dict = self.model(batch)
            output_dict = self.model.discriminate(output_dict, batch, adversarial=True)

            loss = output_dict['loss']
            losses_output_dict['loss'] = float(loss.item())

            loss_D_adv_meaning = output_dict['loss_D_adv_meaning']
            losses_output_dict['loss_D_adv_meaning'] = float(loss_D_adv_meaning.item())
            if self.use_motivator:
                loss_D_adv_style = output_dict['loss_D_adv_style']
                losses_output_dict['loss_D_adv_style'] = float(loss_D_adv_style.item())
            else:
                loss_D_adv_style = 0

            if 'meaning_embedding' in batch and self.use_predictor:
                loss_P_adv_style = output_dict['loss_P_adv_style']
                losses_output_dict['loss_P_adv_style'] = float(loss_P_adv_style.item())

                if self.use_motivator:
                    loss_P_adv_meaning = output_dict['loss_P_adv_meaning']
                    losses_output_dict['loss_P_adv_meaning'] = float(loss_P_adv_meaning.item())
                else:
                    loss_P_adv_meaning = 0
            else:
                loss_P_adv_style = 0
                loss_P_adv_meaning = 0

            if 'meaning_bow' in batch and self.use_predictor_bow:
                loss_P_bow_adv_style = output_dict['loss_P_bow_adv_style']
                losses_output_dict['loss_P_bow_adv_style'] = float(loss_P_bow_adv_style.item())

                if self.use_motivator:
                    loss_P_bow_adv_meaning = output_dict['loss_P_bow_adv_meaning']
                    losses_output_dict['loss_P_bow_adv_meaning'] = float(loss_P_bow_adv_meaning.item())
                else:
                    loss_P_bow_adv_meaning = 0
            else:
                loss_P_bow_adv_style = 0
                loss_P_bow_adv_meaning = 0

            if self.use_gauss:
                G_logits = self.model.D_hidden(output_dict['decoder_hidden'])
                loss_D_adv_hidden = self.model._D_adv_loss(G_logits)
                losses_output_dict['loss_D_adv_hidden'] = float(loss_D_adv_hidden.item())
            else:
                loss_D_adv_hidden = 0

            loss_total = loss
            if loss_D_meaning <= 0.35:
                loss_total += self.D_loss_multiplier * loss_D_adv_meaning
            if loss_D_style <= 0.35:
                loss_total += self.D_loss_multiplier * loss_D_adv_style

            if loss_P_style < 2.5e-3:
                loss_total += self.P_loss_multiplier * loss_P_adv_style
            if loss_P_meaning < 2.5e-3:
                loss_total += self.P_loss_multiplier * loss_P_adv_meaning

            if loss_P_bow_meaning <= 0.35:
                loss_total += self.P_bow_loss_multiplier * loss_P_bow_adv_meaning
            if loss_P_bow_style <= 0.35:
                loss_total += self.P_bow_loss_multiplier * loss_P_bow_adv_style

            if loss_D_hidden <= 0.35:
                loss_total += self.D_loss_multiplier * loss_D_adv_hidden

            losses_output_dict['loss_total'] = float(loss_total.item())

            if self.training:
                loss_total.backward()

                if self.grad_clipping != 0:
                    torch.nn.utils.clip_grad_norm_(self.params_encoder_decoder, self.grad_clipping)

                self.optimizer_encoder_decoder.step()
                self.model.zero_grad()

        return losses_output_dict


class StyleClassifierUpdateFunction(UpdateFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.params = self.get_parameters(self.model)
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)

    def step(self, engine, batch):
        self.optimizer.zero_grad()

        output_dict = self.model(batch)

        loss = output_dict["loss"]

        if self.training:
            loss.backward()

            if self.grad_clipping != 0:
                torch.nn.utils.clip_grad_norm_(self.params, self.grad_clipping)

            self.optimizer.step()

        output_dict = {
            'loss': loss,
        }
        return output_dict
