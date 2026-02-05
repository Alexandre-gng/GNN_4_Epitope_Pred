import torch

"""
Binary Cross Entropy with focal loss class allows the model to focus on the hard predictions and less on the evident one. More precisely 
it decreases the model loss function on the easy predictions (non-epitope) while increasing the loss for hard prediction (epitopes)

In our context of a highly imbalanced dataset (1.6% of epitopes), focussing on hard example seems to be useful.

There is also the Alpha factor that helps the loss function to deal with the imbalanced dataset, it decreases the loss function for the 
majority class.
"""
class FocalLoss(torch.nn.Module):
    """
    Custom Focal Loss function

    "Calibrating Deep Neural Networks using Focal Loss", Mukhoti et al., 2020
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs : logits (non passés dans une sigmoïde)
        targets : étiquettes binaires (0 ou 1)
        """
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)

        # p_t = probas correctes
        pt = torch.where(targets == 1, probs, 1 - probs)

        # α_t dépend de la classe
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Terme focal
        focal_factor = (1 - pt) ** self.gamma

        loss = alpha_t * focal_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
