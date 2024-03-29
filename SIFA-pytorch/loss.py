# loss function for SIFA
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def dice_loss(predict, target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict * target)
    dice = (2 * intersect + smooth) / (torch.sum(target) + torch.sum(predict * predict) + smooth)
    loss = 1.0 - dice
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = ((input_tensor == i).float() * torch.ones_like(input_tensor)).float()
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target, weight=None, softmax=True):
        if softmax:
            inputs = F.softmax(input, dim=1)
        target = self.one_hot_encode(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        return loss / self.n_classes


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            # print("weightc for c",c,weight_c)
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device) + 1e-5
        weight = 1 - weight / (torch.sum(weight)) + 1e-5
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        # print("Weight CE:",weight)
        return wce_loss


class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, num_classes, alpha=1.0):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label)
        celoss = self.celoss(predict, label)
        loss = celoss + self.alpha * diceloss
        print(celoss, "celoss")
        print(diceloss, "diceloss")

        return loss
