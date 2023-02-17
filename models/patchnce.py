from packaging import version
import torch
from torch import nn

"""
Note this is different from the vanilla CUT. I use cosine similarity here,
but both implementations are identical (after the L2 normalization).
You may choose class PatchNCEloss2 (used in vanilla CUT, no cosine similarity), by replacing line 5 in ./dcl_model.py
or ./simdcl_model.py with "from .patchnce import PatchNCELoss2".
"""


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q, feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize, 1, -1), feat_k.view(1, batchSize, -1))
        l_neg_curbatch = l_neg_curbatch.view(1, batchSize, -1)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss


# Used in vanilla CUT
class PatchNCELoss2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        # bmm: batch matrix-matrix product of matrices
        # torch.Size([512, 256]) -> torch.Size([512, 1, 256]), torch.Size([512, 256, 1]) -> torch.Size([512, 1, 1])
        # i.e., sampling B pos-neg patch pairs
        # l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # l_pos = l_pos.view(batchSize, 1)
        l_pos = torch.einsum('BD,BD->B', feat_q, feat_k).unsqueeze(1)  # equivalent implementation

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)

        # torch.Size([2, 256, 256]) torch.Size([2, 256, 256]) -> torch.Size([2, 256, 256])
        # 512 patches are composed of two parts, sampling 256 patches from each image
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # fills elements of self tensor with value where mask is True
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # torch.Size([512, 1]) torch.Size([512, 256]) -> torch.Size([512, 257]), i.e., BxC
        # nce_T, i.e., tau
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # logit at position 0 is positive
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


# Using smooth CE loss in CUT
class PatchNCELossWithLabelSmooth(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = LabelSmoothSoftmaxCEV1(lb_smooth=0.1, reduction='reduction')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        # bmm: batch matrix-matrix product of matrices
        # torch.Size([512, 256]) -> torch.Size([512, 1, 256]), torch.Size([512, 256, 1]) -> torch.Size([512, 1, 1])
        # i.e., sampling B pos-neg patch pairs
        # l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # l_pos = l_pos.view(batchSize, 1)
        l_pos = torch.einsum('BD,BD->B', feat_q, feat_k).unsqueeze(1)  # equivalent implementation

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)

        # torch.Size([2, 256, 256]) torch.Size([2, 256, 256]) -> torch.Size([2, 256, 256])
        # 512 patches are composed of two parts, sampling 256 patches from each image
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # fills elements of self tensor with value where mask is True
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # torch.Size([512, 1]) torch.Size([512, 256]) -> torch.Size([512, 257]), i.e., BxC
        # nce_T, i.e., tau
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # logit at position 0 is positive
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


# Using smooth CE loss in CUT
class PatchNCELossWithFocal(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = FocalLossV1(alpha=0.25, gamma=2, reduction='reduction')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        # bmm: batch matrix-matrix product of matrices
        # torch.Size([512, 256]) -> torch.Size([512, 1, 256]), torch.Size([512, 256, 1]) -> torch.Size([512, 1, 1])
        # i.e., sampling B pos-neg patch pairs
        # l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # l_pos = l_pos.view(batchSize, 1)
        l_pos = torch.einsum('BD,BD->B', feat_q, feat_k).unsqueeze(1)  # equivalent implementation

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)

        # torch.Size([2, 256, 256]) torch.Size([2, 256, 256]) -> torch.Size([2, 256, 256])
        # 512 patches are composed of two parts, sampling 256 patches from each image
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # fills elements of self tensor with value where mask is True
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # torch.Size([512, 1]) torch.Size([512, 256]) -> torch.Size([512, 257]), i.e., BxC
        # nce_T, i.e., tau
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # logit at position 0 is positive
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/label_smooth.py
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


# Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
class PatchNCELossDCL(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        # bmm: batch matrix-matrix product of matrices
        # torch.Size([512, 256]) -> torch.Size([512, 1, 256]), torch.Size([512, 256, 1]) -> torch.Size([512, 1, 1])
        # i.e., sampling B pos-neg patch pairs
        # l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # l_pos = l_pos.view(batchSize, 1)
        l_pos = torch.einsum('BD,BD->B', feat_q, feat_k).unsqueeze(1)  # equivalent implementation

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)

        # torch.Size([2, 256, 256]) torch.Size([2, 256, 256]) -> torch.Size([2, 256, 256])
        # 512 patches are composed of two parts, sampling 256 patches from each image
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # fills elements of self tensor with value where mask is True
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # torch.Size([512, 1]) torch.Size([512, 256]) -> torch.Size([512, 257]), i.e., BxC
        # nce_T, i.e., tau
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # logit at position 0 is positive
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss


import numpy as np

SMALL_NUM = np.log(1e-45)


class DCL(nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super().__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def _one_way_loss(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        positive_loss = -(z1 * z2).sum(dim=1) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), torch.mm(z1, z2.t())), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()

    def forward(self, z1, z2):
        return self._one_way_loss(z1, z2) + self._one_way_loss(z2, z1)


class DCLW(DCL):
    """
    Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
    sigma: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, sigma=0.5, temperature=0.1):
        weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma,
                                                                                dim=0).squeeze()
        super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)


# Used in CUT++
class StylePatchNCELoss2(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, feat_v):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        feat_v = feat_v.detach()

        # pos logit
        # bmm: batch matrix-matrix product of matrices
        # torch.Size([512, 256]) -> torch.Size([512, 1, 256]), torch.Size([512, 256, 1]) -> torch.Size([512, 1, 1])
        # i.e., sampling B pos-neg patch pairs
        # l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        # l_pos = l_pos.view(batchSize, 1)
        l_pos = torch.einsum('BD,BD->B', feat_q, feat_v).unsqueeze(1)  # equivalent implementation

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)

        # torch.Size([2, 256, 256]) torch.Size([2, 256, 256]) -> torch.Size([2, 256, 256])
        # 512 patches are composed of two parts, sampling 256 patches from each image
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        # fills elements of self tensor with value where mask is True
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # torch.Size([512, 1]) torch.Size([512, 256]) -> torch.Size([512, 257]), i.e., BxC
        # nce_T, i.e., tau
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # logit at position 0 is positive
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss
