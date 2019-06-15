import torch
import torch.nn.functional as F


class SequenceReconstructionLoss(torch.nn.Module):
    def __init__(self, ignore_index=-100):
        super(SequenceReconstructionLoss, self).__init__()

        self.xent_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _calc_sent_xent(self, outputs, targets):
        if len(outputs.shape) > 2:
            targets = targets.view(-1)
            outputs = outputs.view(targets.size(0), -1)

        xent = self.xent_loss(outputs, targets)

        return xent

    def forward(self, outputs, targets):
        loss = self._calc_sent_xent(outputs, targets)

        return loss


class StyleEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(StyleEntropyLoss, self).__init__()

        self.epsilon = 1e-07

    def forward(self, logits):
        probs = torch.sigmoid(logits)
        entropy = probs * torch.log(probs + self.epsilon) + (1 - probs) * torch.log(1 - probs + self.epsilon)
        entropy = torch.mean(entropy, dim=-1)

        loss_mean = torch.mean(entropy)  # No `-1 *` as we are going to add it to the loss

        return loss_mean


class MeaningZeroLoss(torch.nn.Module):
    def __init__(self):
        super(MeaningZeroLoss, self).__init__()

    def forward(self, predicted):
        loss = predicted ** 2
        loss_mean = torch.mean(loss)

        return loss_mean


class LSGANDiscriminatorLoss(torch.nn.Module):
    # Least Squares GAN
    def __init__(self):
        super(LSGANDiscriminatorLoss, self).__init__()

    def forward(self, logits, styles):
        logits_zero = logits[styles == 0]
        logits_one = logits[styles == 1]

        loss = 0.5 * (torch.mean((logits_zero - 1) ** 2) + torch.mean(logits_one ** 2))

        return loss
