import os
import piq
import torch
from piq import *
import numpy as np
import random
from itertools import chain
from util.loss import MINDLoss, MILoss

import PIL.Image as Image
from prettytable import PrettyTable

from torchvision.transforms.functional import to_tensor

import os
import os.path as osp
import itertools


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def quality_metric(pred, gt, metric):
    loss_list = []
    assert len(pred) != 0 and len(pred) == len(gt)
    for pred_path, gt_path in zip(pred, gt):
        # img_pred = Image.open(pred_path).convert('L')
        # img_gt = Image.open(gt_path).convert('L')
        img_pred = Image.open(pred_path)
        img_gt = Image.open(gt_path)

        img_pred = to_tensor(img_pred).unsqueeze(0)
        img_gt = to_tensor(img_gt).unsqueeze(0)

        loss = metric(img_pred, img_gt)

        loss_list.append(loss.item())

    return sum(loss_list) / len(loss_list), loss_list


metric_no_ref = ['total_variation', 'brisque', 'inception_score']


def quality_metric_no_ref(pred, gt, metric):
    loss_list_a = []
    loss_list_b = []
    assert len(pred) != 0 and len(pred) == len(gt)
    for pred_path, gt_path in zip(pred, gt):
        # img_pred = Image.open(pred_path).convert('L')
        # img_gt = Image.open(gt_path).convert('L')
        img_pred = Image.open(pred_path)
        img_gt = Image.open(gt_path)

        img_pred = to_tensor(img_pred).unsqueeze(0)
        img_gt = to_tensor(img_gt).unsqueeze(0)

        loss_a = metric(img_pred)
        loss_b = metric(img_gt)
        loss_list_a.append(loss_a.item())
        loss_list_b.append(loss_b.item())

    return (sum(loss_list_a) / len(loss_list_a), sum(loss_list_b) / len(loss_list_b)), (loss_list_a, loss_list_b)


def eval_metric(fake_path, real_path, metric):
    fake_list = []
    real_list = []

    # print(len(os.listdir(fake_path)))

    for img_name in os.listdir(fake_path):
        if img_name.endswith('.png'):
            fake_list.append(osp.join(fake_path, img_name))
            real_list.append(osp.join(real_path, img_name))

    if metric.__name__ not in metric_no_ref:
        print(metric.__name__)
        mean_loss, loss_list = quality_metric(fake_list, real_list, metric)
    else:
        print(metric.__name__)
        mean_loss, loss_list = quality_metric_no_ref(fake_list, real_list, metric)

    # print(mean_loss)  # 0.6306624141335487
    return mean_loss, loss_list


def mind(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        loss = MINDLoss(*args).to('cuda:0')
    else:
        loss = MINDLoss(*args)
    return 100 * loss(x, y)


def mutual_info(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        loss = MILoss(*args).to('cuda:0')
    else:
        loss = MILoss(*args)
    return loss(x, y)


def dists(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
    loss = piq.DISTS(*args)
    return loss(x, y)


def lpips(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
    loss = piq.LPIPS(*args)
    return loss(x, y)


def pieapp(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
    loss = piq.PieAPP(*args)
    return loss(x, y)


def basic():
    import os

    # train
    # with batch_size = 4, take ~50 hrs
    # sh = "python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan --batch_size 2"

    dataset = "brain_mr2ct"  # mr2ct
    # dataset = "retina2vessel"  # mr2ct
    # dataset = "oneshot"  # mr2ct
    # dataset = "horse2zebra"  # mr2ct

    name = "paired"

    sh = f"python train.py --dataroot ./datasets/{dataset} --name {dataset}_{name}"
    paras = [
        " --model pix2pix",  # cycle_gan,
        # " --direction BtoA",
        " --n_epochs 100",
        " --n_epochs_decay 100",
        " --gan_mode lsgan",  # vanilla, lsgan, lsgan+mind
        " --batch_size 4",
        # " --netG unet_256",
        # " --ngf 96",
        " --single_D True",
        # " --single_G True",
        # " --netD pixel",  # pixel
        # " --n_layers_D 5",
        # " --lambda_identity 0 --display_ncols 3",

        # " --cycle_loss SSMI",
        # " --aug_policy color,cutout",  # color,translation,cutout
        # " --aug_threshold 0.5",  # 1 - p
        # " --preprocess scale_width_and_crop --num_threads 0",
        # " --align_mode DISTS",
        # " --lambda_DISTS 0.5",

        # " --align_mode MIND",
        # " --lambda_MIND 0",

        # " --align_mode MI",
        # " --lambda_MI 0.1",

        # " --lambda_A 10",
        # " --lambda_B 10",
        " --save_epoch_freq 50",
        " --load_size 256",
        " --crop_size 256",
        " --input_nc 1",
        " --output_nc 1",
    ]

    for x in paras:
        sh += x

    # os.system(sh)

    # test
    # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
    sh = f"python test.py --name {dataset}_{name} --model test --no_dropout"
    #
    paras = [
        f" --dataroot datasets/{dataset}",
        " --model pix2pix",  # cycle_gan,
        # " --direction BtoA",
        # " --netG unet_256",
        " --load_size 256",
        # " --crop_size 512",
        " --num_test 200"
        " --input_nc 1",
        " --output_nc 1",
    ]
    #
    for x in paras:
        sh += x

    os.system(sh)

    from piq import ssim, psnr

    fake_path = f"results/brain_mr2ct_paired/test_latest/images"
    real_path = f"Z:/GAN/pytorch-CycleGAN-and-pix2pix-master/datasets/brain_mr2ct/allB"

    eval_metric(fake_path, real_path, ssim)


class Pix2Pix:
    def __init__(self, dataset, name, batch_size, gan_mode='lsgan', continue_train=False):
        self.dataset = dataset  # mr2ct
        self.name = name
        self.batch_size = batch_size
        self.gan_mode = gan_mode
        self.continue_train = continue_train

    def train(self):

        sh = f"python train.py --dataroot ./datasets/{self.dataset} --name {self.dataset}_{self.name}"
        paras = [
            " --model pix2pix",  # cycle_gan,
            " --n_epochs 100",
            " --n_epochs_decay 100",
            f" --gan_mode {self.gan_mode}",  # vanilla, lsgan
            f" --batch_size {self.batch_size}",
            f" --continue_train" if self.continue_train else "",

            " --save_epoch_freq 50",
            " --load_size 256",
            " --crop_size 256",
            " --input_nc 1",
            " --output_nc 1",
        ]

        for x in paras:
            sh += x

        os.system(sh)

    def test(self):
        # test
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python test.py --name {self.dataset}_{self.name} --model test --no_dropout"
        #
        paras = [
            f" --dataroot datasets/{self.dataset}",
            " --model pix2pix",  # cycle_gan,
            " --load_size 256",
            " --crop_size 256",
            " --num_test 1000"
            " --input_nc 1",
            " --output_nc 1",
        ]
        #
        for x in paras:
            sh += x

        os.system(sh)

    def eval(self, metric):

        fake_path = f"results/{self.dataset}_{self.name}/test_latest/images"
        real_path = f"Z:/GAN/pytorch-CycleGAN-and-pix2pix-master/datasets/{self.dataset}/allB"

        eval_metric(fake_path=fake_path, real_path=real_path, metric=metric)


class Pix2Pix:
    def __init__(self, dataset, name, model="pix2pix", load_size=256, dataset_mode='alignedslices', netG='unet_256',
                 input_nc=1, output_nc=1,
                 extra=''):
        self.dataset = dataset  # mr2ct
        self.name = name

        self.model = model
        self.netG = netG
        self.load_size = load_size
        self.dataset_mode = dataset_mode
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.shared_args = [
            f" --netG {netG}",
            f" --dataset_mode {self.dataset_mode}",
            extra
        ]

    def train(self, batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_L1=100, input_nc=1,
              nce_idt=True, extra='',
              continue_train=False):
        sh = f"python train.py --dataroot ./datasets/{self.dataset} --name {self.dataset}_{self.name}"
        paras = [
            f" --model {self.model}",  # cycle_gan, simdcl, cut, fastcut
            # " --direction BtoA",
            f" --lambda_L1 {lambda_L1}",
            f" --n_epochs {n_epochs}",
            f" --n_epochs_decay {n_epochs_decay}",
            f" --gan_mode hinge",  # vanilla, lsgan, lsgan+mind
            f" --batch_size {batch_size}",

            f" --load_size {self.load_size}",
            " --crop_size 256",
            f" --input_nc {self.input_nc} --output_nc {self.output_nc}",

            " --num_threads 8",
            " --continue_train" if continue_train else "",
            # f" --lambda_identity {lambda_identity}",
            # " --netG unet_256",
            # " --ngf 96",
            # " --netD pixel",  # pixel
            # " --n_layers_D 5",
            # " --lambda_identity 0 --display_ncols 3" if self.no_identity else "",

            # " --lambda_A 10",
            # " --lambda_B 10",
            " --save_epoch_freq 50",
            " --display_ncols 3",
            extra
        ]

        for x in chain(self.shared_args, paras):
            sh += x

        os.system(sh)

    def test(self, num_test=500, script='test', extra=''):
        # test
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python {script}.py --name {self.dataset}_{self.name} --phase test --no_dropout"
        #
        paras = [
            f" --dataroot datasets/{self.dataset}",
            f" --model {self.model}",
            " --load_size 256",
            # " --crop_size 512",
            f" --num_test {num_test}"
            " --input_nc 1 --output_nc 1",
            extra
        ]
        #
        for x in chain(self.shared_args, paras):
            sh += x

        os.system(sh)

    @torch.no_grad()
    def eval(self, metric_list, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [self.name]
        exp_row_B = [self.name]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return exp_row_A, exp_row_B

    @torch.no_grad()
    def eval_spos(self, metric_list, choice_spos, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B/{choice_spos}"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B/{choice_spos}"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A/{choice_spos}"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A/{choice_spos}"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
            "ssmi": "Similarity",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [choice_spos]
        exp_row_B = [choice_spos]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return mean_loss_A, mean_loss_B

    def fid(self):
        sh = f"python -m pytorch_fid" \
             f" ./results/{self.dataset}_{self.name}/test_latest/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/test_latest/images/real_B"
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/test_latest/images/fake_B")
        os.system(sh)


class CycleGAN:
    def __init__(self, dataset, name, model="dcl", load_size=512, dataset_mode='unaligned', netG='resnet_9blocks',
                 input_nc=1, output_nc=1,
                 extra=''):
        self.dataset = dataset  # mr2ct
        self.name = name

        self.model = model
        self.netG = netG
        self.load_size = load_size
        self.dataset_mode = dataset_mode
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.shared_args = [
            f" --netG {netG}",
            f" --dataset_mode {self.dataset_mode}",
            extra
        ]

    def train(self, batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_identity=0.5,
              nce_idt=True, extra='',
              continue_train=False):
        sh = f"python train.py --dataroot ./datasets/{self.dataset} --name {self.dataset}_{self.name}"
        paras = [
            f" --model {self.model}",  # cycle_gan, simdcl, cut, fastcut
            # " --direction BtoA",
            f" --lambda_identity {lambda_identity}",
            f" --n_epochs {n_epochs}",
            f" --n_epochs_decay {n_epochs_decay}",
            f" --gan_mode hinge",  # vanilla, lsgan, lsgan+mind
            f" --batch_size {batch_size}",

            f" --load_size {self.load_size}",
            " --crop_size 256",
            f" --input_nc {self.input_nc} --output_nc {self.output_nc}",

            " --num_threads 8",
            " --continue_train" if continue_train else "",
            # f" --lambda_identity {lambda_identity}",
            # " --netG unet_256",
            # " --ngf 96",
            # " --netD pixel",  # pixel
            # " --n_layers_D 5",
            # " --lambda_identity 0 --display_ncols 3" if self.no_identity else "",

            # " --lambda_A 10",
            # " --lambda_B 10",
            " --save_epoch_freq 50",
            " --display_ncols 3",
            extra
        ]

        for x in chain(self.shared_args, paras):
            sh += x

        os.system(sh)

    def test(self, num_test=500, script='test', extra=''):
        # test
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python {script}.py --name {self.dataset}_{self.name} --phase test --no_dropout"
        #
        paras = [
            f" --dataroot datasets/{self.dataset}",
            f" --model {self.model}",
            " --load_size 256",
            # " --crop_size 512",
            f" --num_test {num_test}"
            " --input_nc 1 --output_nc 1",
            extra
        ]
        #
        for x in chain(self.shared_args, paras):
            sh += x

        os.system(sh)

    @torch.no_grad()
    def eval(self, metric_list, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [self.name]
        exp_row_B = [self.name]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return exp_row_A, exp_row_B

    @torch.no_grad()
    def eval_spos(self, metric_list, choice_spos, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B/{choice_spos}"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B/{choice_spos}"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A/{choice_spos}"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A/{choice_spos}"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
            "ssmi": "Similarity",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [choice_spos]
        exp_row_B = [choice_spos]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return mean_loss_A, mean_loss_B

    def fid(self):
        sh = f"python -m pytorch_fid" \
             f" ./results/{self.dataset}_{self.name}/test_latest/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/test_latest/images/real_B"
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/test_latest/images/fake_B")
        os.system(sh)


class Experiment:
    def __init__(self, dataset, name, model="dcl", load_size=256, netG="resnet_9blocks",
                 input_nc=1, output_nc=1, dataroot='./datasets', extra="",
                 dataset_mode='unaligned', gpu_ids='0'):
        self.dataset = dataset  # mr2ct
        self.name = name
        self.model = model
        self.load_size = load_size
        self.dataset_mode = dataset_mode
        self.dataroot = dataroot

        self.shared_args = [
            f" --input_nc {input_nc} --output_nc {output_nc}",
            f" --netG {netG}",
            f" --dataroot {self.dataroot}/{self.dataset}",
            f" --dataset_mode {self.dataset_mode}",
            f" --gpu_ids {gpu_ids}",

            extra,
        ]

    def train(self, batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_identity=0.5, input_nc=1, output_nc=1,
              nce_idt=True, display_ncols=3,
              continue_train=False, script_name='train', extra=""):
        sh = f"python {script_name}.py --name {self.dataset}_{self.name}"
        paras = [
            f" --model {self.model}",  # cycle_gan, simdcl, cut, fastcut
            # " --direction BtoA",
            f" --nce_idt {nce_idt}" if nce_idt is not None else "",
            f" --n_epochs {n_epochs}",
            f" --n_epochs_decay {n_epochs_decay}",
            f" --gan_mode hinge",  # vanilla, lsgan, lsgan+mind
            f" --batch_size {batch_size}",

            f" --load_size {self.load_size}",
            " --crop_size 256",

            " --num_threads 8",
            " --continue_train" if continue_train else "",
            # f" --lambda_identity {lambda_identity}",
            # " --netG unet_256",
            # " --ngf 96",
            # " --netD pixel",  # pixel
            # " --n_layers_D 5",
            # " --lambda_identity 0 --display_ncols 3" if self.no_identity else "",

            # " --lambda_A 10",
            # " --lambda_B 10",
            " --save_epoch_freq 50",
            f" --display_ncols {display_ncols}",
        ]

        for x in chain(self.shared_args, paras):
            sh += x

        sh += extra
        print(sh)
        os.system(sh)

    def test(self, script='test_3d', num_test=500, phase='test', extra=''):
        # test
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python {script}.py --name {self.dataset}_{self.name} --phase {phase} --no_dropout"
        #
        paras = [
            f" --dataset_mode {self.dataset_mode}",
            f" --model {self.model}",
            " --load_size 256",
            # " --crop_size 512",
            f" --num_test {num_test}",
            extra
        ]
        #
        for x in chain(self.shared_args, paras):
            sh += x

        os.system(sh)

    @torch.no_grad()
    def eval(self, metric_list, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
            "ssmi": "Similarity",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [self.name]
        exp_row_B = [self.name]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return exp_row_A, exp_row_B

    @torch.no_grad()
    def eval_spos(self, metric_list, choice_spos, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B/{choice_spos}"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B/{choice_spos}"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A/{choice_spos}"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A/{choice_spos}"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [choice_spos]
        exp_row_B = [choice_spos]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return mean_loss_A, mean_loss_B

    def fid(self):
        sh = f"python -m pytorch_fid" \
             f" ./results/{self.dataset}_{self.name}/test_latest/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/test_latest/images/real_B"
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/test_latest/images/fake_B")
        os.system(sh)


def show_results(metric_list, rowsA, rowsB, testB=True):
    # table = PrettyTable(title=exp.dataset, field_names=['metric', 'testA', 'testB'])
    field_names = ['method']
    for metric in metric_list:
        field_names.append(metric)
    tableA = PrettyTable(title="testA", field_names=field_names)
    tableB = PrettyTable(title="testB", field_names=field_names)

    for rowA, rowB in zip(rowsA, rowsB):
        tableA.add_row(rowA)
        tableB.add_row(rowB)

    print(tableA)
    if testB:
        print(tableB)


def exp_bk():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #
    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="cyclegan", name="cyclegan_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="cut", name="cut_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="ct2mr", model="cut", name="cut_seq", load_size=256, dataset_mode='unalignedseq')
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # # ##############################  ct2mr  ############################### # #

    # cut_seq, 5s
    exp = Experiment(dataset="ct2mr", model="cut", name="cut_seq", load_size=256, dataset_mode='unalignedseq')
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # # ###########################  retina2vessel  ############################ # #

    # dcl, 8s
    exp = Experiment(dataset="retina2vessel", model="dcl", name="dcl")
    # exp.train(batch_size=2, n_epochs=400, n_epochs_decay=1600, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=True)

    # cut, 5s
    exp = Experiment(dataset="retina2vessel", model="cut", name="cut")
    # exp.train(batch_size=2, n_epochs=400, n_epochs_decay=1600, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # cutpp, 5s
    exp = Experiment(dataset="retina2vessel", model="cutpp", name="cutpp")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=2000, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # cyclegan, 5s
    # exp = Experiment(dataset="retina2vessel", model="cycle_gan", name="cyclegan", load_size=256)
    # exp.train(batch_size=2, n_epochs=400, n_epochs_decay=1600, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.eval(metric_list, testB=True)

    show_results(metric_list, rowsA, rowsB)


def legacy():
    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd3d", name="cutnd3d_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices2n',
                     extra=" --num_K 3 --ngf 8 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")  # --netD basic3d
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)


def livecell():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    exp = Experiment(dataset="LIVECell", model="cut", name="cut_ep200", load_size=256,
                     input_nc=3, output_nc=3,
                     netG='hrt_nb', extra=' --n_blocks 2 --nce_layers 0,4,8,12 --ndf 48',
                     dataset_mode="unaligned4is")
    exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=3, output_nc=3, nce_idt=True,
              continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def baseline_IXI():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # # ##############################  IXI  ############################### # #

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd", name="cutnd_ep10", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 1 --ngf 64")
    exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
              extra=" --save_latest_freq 5000 --display_ncols=4")  # --netD basic3d
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)


def MR2CT_Reg():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # # ##############################  IXI  ############################### # #

    # 1840 sec/epoch
    exp = Experiment(dataset="original_TRSAA_crop_BY_TUMOR", model="cutndv2", name="cutndv2_ep5k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4noreg', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 1 --ngf 64")
    exp.train(batch_size=1, n_epochs=2500, n_epochs_decay=2500, nce_idt=True, continue_train=False,
              extra=" --save_latest_freq 5000 --display_ncols=4 --display_freq 100")  # --netD basic3d
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)


def BraTS19():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # # ##############################  IXI  ############################### # #

    exp = Experiment(dataset="original_TRSAA_crop_BY_TUMOR", model="dcl3d", name="dcl3d_ep2k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4seg176x2', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    exp.train(batch_size=2, n_epochs=1000, n_epochs_decay=1000, nce_idt=True, continue_train=False,
              extra=" --save_latest_freq 5000 --display_freq 50"
                    " --display_ncols 7 --eval_metric --eval_freq 200")  # --netD basic3d
    # exp.test(phase='train')
    # exp.test(phase='test')
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS19", model="dcl3d19v2", name="dcl3d19v2_ep3k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unaligned4brats19', gpu_ids='0',
                     dataroot="/media/cas/4053447d-1eaa-4b32-ad96-a8c03e4e35d2/DataBaseNo.1",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=2, n_epochs=1500, n_epochs_decay=1500, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_freq 50"
    #                 " --display_ncols 7 --eval_metric --eval_freq 200")  # --netD basic3d

    # 1840 sec/epoch
    exp = Experiment(dataset="MMSeg", model="mmseg", name="mmseg_ep3k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='mmseg', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=1, n_epochs=1500, n_epochs_decay=1500, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_freq 50"
    #                 " --display_ncols 7 --eval_metric --eval_freq 20")  # --netD basic3d

    # 1840 sec/epoch
    exp = Experiment(dataset="original_TRSAA_crop", model="dcl3d", name="dcl3d_ep2k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4seg176x2', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=2, n_epochs=1000, n_epochs_decay=1000, nce_idt=True, continue_train=True,
    #           extra=" --save_latest_freq 5000 --display_freq 50"
    #                 " --display_ncols 7 --eval_metric --eval_freq 200")  # --netD basic3d
    # exp.test(phase='train')
    # exp.test(phase='test')
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="BraTS19", model="dcl3d19", name="dcl3d_ep3k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unaligned4brats19', gpu_ids='0',
                     dataroot="/media/cas/4053447d-1eaa-4b32-ad96-a8c03e4e35d2/DataBaseNo.1",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=2, n_epochs=1500, n_epochs_decay=1500, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_freq 50"
    #                 " --display_ncols 7 --eval_metric --eval_freq 200")  # --netD basic3d

    # 1840 sec/epoch
    exp = Experiment(dataset="original_TRSAA_crop", model="dcl3dv2", name="dcl3dv2_ep3k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4seg176x2', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=2, n_epochs=1500, n_epochs_decay=1500, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_freq 50"
    #                 " --display_ncols 6 --eval_metric --eval_freq 200")  # --netD basic3d


    # 1840 sec/epoch
    exp = Experiment(dataset="BraTS19", model="tricycle", name="tricycle_ep5k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unaligned4brats19', gpu_ids='0',
                     dataroot="/media/cas/4053447d-1eaa-4b32-ad96-a8c03e4e35d2/DataBaseNo.1",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=4, n_epochs=2500, n_epochs_decay=2500, nce_idt=False, continue_train=True,
    #           extra=" --save_latest_freq 5000 --display_freq 50"
    #                 " --display_ncols 6 --eval_metric --eval_freq 200")  # --netD basic3d
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)


def baseline():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #

    exp = Experiment(dataset="IXI", model="cut", name="cutswin_ep10", load_size=256, netG='swin',
                     extra='')  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutcam", name="cutcam_ep10", load_size=256, netG='resnet_9blocks',
                     extra=' --netF mlp_cam_sample')  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False, extra=' --gan_mode lsgan')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutnd", name="cut3d_hgk_ep10", load_size=256, netG='hrkormer',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 1 --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv5_ep10", load_size=256, netG='hrkormerv5',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv2_ep10", load_size=256, netG='hrkormerv2',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14 --lambda_NCE 2")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv3_ep10", load_size=256, netG='hrkormerv3',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14 --lambda_NCE 2")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv4_ep10", load_size=256, netG='hrkormerv4',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14 --lambda_NCE 2")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="horse2zebra", model="cut", name="cut_hgk_ep200", load_size=256, netG='hrkormer',
                     input_nc=3, output_nc=3,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="cycle_gan", name="cyclegan_hrkormer_ep200", load_size=256,
                   netG='hrkormer',
                   input_nc=3, output_nc=3)
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutcam", name="cutcam_ep10", load_size=256, netG='resnet_9blocks',
                     extra=' --netF mlp_cam_sample')  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False, extra=' --gan_mode lsgan')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # # UNet
    # # 1546 sec, ~0.5h ## 04789
    # exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_vip_ep10", load_size=256, netG='VIP')
    # exp.train(batch_size=1, n_epochs=0, n_epochs_decay=8, lambda_identity=0, continue_train=True)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # UNet
    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_unet256_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cycle_gan3d_unet256_ep10", load_size=256, netG='unet_256',
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutpiq", name="cutpiq_ms-ssim_e-1_ep10", load_size=256,
                     netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=' --piq_name MS-SSIM --lambda_PIQ 0.1')
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutpiq", name="cutpiq_ms-gmsd_e-1_ep10", load_size=256,
                     netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=' --piq_name MS-GMSD --lambda_PIQ 0.1')
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutpiq", name="cutpiq_ms-gmsd_5e0_ep10", load_size=256,
                     netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=' --piq_name MS-GMSD --lambda_PIQ 0.5')
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutsce", name="cutsce_e-1_ep10", load_size=256, netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutfocal", name="cutfocal_ep10", load_size=256, netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256, netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    #

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def baseline_cam():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #

    exp = Experiment(dataset="IXI", model="cut2f", name="cut2f_hrt_ep20", load_size=256, netG='hrt_nb',
                     extra=' --n_blocks 6 --nce_layers 0,4,8,12,13,14', input_nc=1, output_nc=1)
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=15, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="cat2dog", model="cutcam", name="cutcam_ep10", load_size=256, netG='resnet_9blocks',
                     extra=' --netF mlp_cam_sample', input_nc=3, output_nc=3)  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=' --gan_mode hinge --netD hg --display_freq 100')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)
    #
    # exp = Experiment(dataset="IXI", model="cutcam", name="cutcam_ep10", load_size=256, netG='resnet_9blocks',
    #                  extra=' --netF mlp_cam_sample')  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=' --gan_mode hinge --netD hg --display_freq 100')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cutswin_ep10", load_size=256, netG='swin',
                     extra='')  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutcam", name="cutcam_ep10", load_size=256, netG='resnet_9blocks',
                     extra=' --netF mlp_cam_sample')  # --nce_layers 2,5,9,12,16
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False, extra=' --gan_mode lsgan')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutnd", name="cut3d_hgk_ep10", load_size=256, netG='hrkormer',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 1 --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv5_ep10", load_size=256, netG='hrkormerv5',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv2_ep10", load_size=256, netG='hrkormerv2',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14 --lambda_NCE 2")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hgkv3_ep10", load_size=256, netG='hrkormerv3',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --n_blocks 6 --nce_layers 0,3,6,10,12,14 --lambda_NCE 2")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256, netG='resnet_9blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    #

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def LPTN():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #
    # UNet
    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_lptn_ep10", load_size=256, netG='lptn')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cycle_gan3d_unet256_ep10", load_size=256, netG='unet_256',
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    #

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def DINTS():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #
    exp = Experiment(dataset="IXI", model="cut", name="cut_mposv1_ep10", load_size=256, netG='mposv1',
                     extra=f" --nce_layers 1,2,3,4,5,6")
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
              display_ncols=4, extra='')
    exp.test()
    exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_mposv1_ep15", load_size=256, netG='mposv1',
                     extra=f" --nce_layers 1,2,3,4,5,6")
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=10, nce_idt=True, continue_train=False,
              display_ncols=4, extra='')
    exp.test()
    exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dints_cut_mposv2", name="dints_cut_mposv2_ep10", load_size=256, netG='spos',
                     extra=f" --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=30, nce_idt=True, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.eval(metric_list, testB=False)
    #
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, input_nc=1, nce_idt=False, continue_train=True, display_ncols=4,
    #           extra=' --DECODE')
    # exp.test(extra=' --DECODE')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dints_cut_mpos", name="dints_cut_norm_ep10", load_size=256, netG='spos',
                     extra=f" --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dints_cut_mpos", name="dints_cut_norm_ep10_widt", load_size=256, netG='spos',
                     extra=f" --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.eval(metric_list, testB=False)
    #

    exp = Experiment(dataset="IXI", model="dints_cut_mpos", name="dints_cut_mpos_ep20", load_size=256, netG='spos',
                     extra=f" --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=20, nce_idt=False, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.eval(metric_list, testB=False)
    #
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, input_nc=1, nce_idt=False, continue_train=True, display_ncols=4,
    #           extra=' --DECODE')
    # exp.test(extra=' --DECODE')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dints_cut_mpos", name="dints_cut_mpos_ep10", load_size=256, netG='spos',
                     extra=f" --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=20, input_nc=1, nce_idt=True, continue_train=True,
    #           display_ncols=4)
    # exp.test()
    # exp.test(extra=' --DECODE')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # UNet
    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="dints_cycle_gan", name="dints_cycle_gan_ep10", load_size=256,
                   dataset_mode='unaligned4nas')
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=0, input_nc=1, lambda_identity=0, continue_train=True,
    #           extra='')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="dints_cycle_gan", name="dints_cycle_gan_ep10", load_size=256, input_nc=1,
                   dataset_mode='unaligned4nas')
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=30, lambda_identity=0, continue_train=False,
    #           extra=' --save_latest_freq 50')  #  --save_latest_freq 200
    # exp.test()  # test with full arch
    # exp.eval(metric_list, testB=False)
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=0, input_nc=1, lambda_identity=0, continue_train=False,
    #           extra=' --DECODE --dataset_mode unaligned')  # retrain
    # exp.test(extra=' --DECODE --dataset_mode unaligned')   # test with full arch
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="dints_cycle_gan_mpos", name="dints_cycle_gan_mpos_ep10", load_size=256,
                   input_nc=1)
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=30, lambda_identity=0.5, continue_train=False,
    #           extra='   ')  #  --save_latest_freq 200
    # exp.test()  # test with full arch
    # exp.eval(metric_list, testB=False)
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=0, input_nc=1, lambda_identity=0, continue_train=False,
    #           extra=' --DECODE --dataset_mode unaligned')  # retrain
    # exp.test(extra=' --DECODE --dataset_mode unaligned')   # test with full arch
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cycle_gan3d_unet256_ep10", load_size=256, netG='unet_256',
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256, netG='unet_256')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    #

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep400", load_size=256)
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def baseline_brats():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #

    # >>>>>>>>>> nce layers

    exp = Experiment(dataset="BraTS18", model="cut", name="cut_ep10", load_size=256, extra=' --nce_layers 0,4,8,12,16')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)
    # Instance Norm
    exp = Experiment(dataset="BraTS18", model="cut", name="cut_ep10", load_size=256, extra=' --nce_layers 2,5,9,12,16')
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)
    # ReLU
    exp = Experiment(dataset="BraTS18", model="cut", name="cut_ep10", load_size=256, extra=' --nce_layers 3,6,10,12,16')
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # >>>>>>>>>> nce layers

    exp = Experiment(dataset="BraTS18", model="cutnd", name="cut3d_hrt_ep10", load_size=256,
                     netG='hrt_nb', extra=' --n_blocks 6 --nce_layers 0,4,8,13',
                     dataset_mode="unalignedslices")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cutnd", name="cut3d_ep10", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cutnd", name="cut3d_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="BraTS18", model="unitv2", name="unitv2_ep10", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="BraTS18", model="unit", name="unit_ep10", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="BraTS18", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cut", name="cut_ep10", load_size=256, )
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cut3d", name="cut3d_ep10", load_size=256,
                     dataset_mode="unaligned4brats")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cut3d", name="cut3d_ep10_widt", load_size=256,
                     dataset_mode="unaligned4brats")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def baseline_ct2mr():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    exp = Experiment(dataset="ct2mr", model="cutc1d", name="cutc1d_ep10", load_size=256, )
    # exp.train(batch_size=3, n_epochs=5, n_epochs_decay=588, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutcld", name="cutcld_ep10", load_size=256, )
    # exp.train(batch_size=3, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=True,
    #           extra=' --netD n_layers_contrast')
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutmsd", name="cutmsd_ep10", load_size=256, )
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=' --netD n_layers_ms')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutmsd", name="cutmsd_ep20", load_size=256, )
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=' --netD n_layers_ms')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutnd", name="cut3d_hrt6_ep400", load_size=256, netG='hrt_nb',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64 --n_blocks 6 --nce_layers 3,6,10,12,14")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutnd", name="cut3d_hrt6_ep200", load_size=256, netG='hrt_nb',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64 --n_blocks 6 --nce_layers 1,4,8,12,14")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutnd", name="cut3d_hrt6_ep200", load_size=256, netG='hrt_nb',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64 --n_blocks 6 --nce_layers 3,6,10,12,14")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutnd", name="cut3d_p2p_ep10", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedp2p',
                     extra=" --num_K 1 --ngf 64")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False, script_name='train_p2p',
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutnd", name="cut3d_ep200", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 1 --ngf 64")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutnd", name="cut3d_ep200", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols 4")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutdiffaug", name="cutdiffaug_ep200", load_size=256, )
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=" --policy color")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cut", name="cut_ep200", load_size=256, )
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #

    # 1840 sec/epoch
    exp = Pix2Pix(dataset="ct2mr", model="pix2pix", name="pix2pix_unet256_ep200", load_size=256, netG="unet_256",
                  input_nc=1, output_nc=1, dataset_mode='alignedslices',
                  extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="ct2mr", model="unit", name="unit_ep400", load_size=256, input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="ct2mr", model="cycle_gan", name="cyclegan_ep200", load_size=256, input_nc=1)
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutsce", name="cutsce_ep200", load_size=256, )
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="ct2mr", model="cutfocal", name="cutfocal_ep200", load_size=256, )
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cut3d", name="cut3d_ep10", load_size=256,
                     dataset_mode="unaligned4brats")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="cut3d", name="cut3d_ep10_widt", load_size=256,
                     dataset_mode="unaligned4brats")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def baseline_horse():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_001000_ep200", load_size=256, netG='spos',
                     input_nc=3, output_nc=3,
                     extra=" --ngf 64 --n_blocks 6 --n_choices 3 --nce_layers 0,4,8,12,14")

    choice = str([0, 0, 1, 0, 0, 0]).replace(' ', '')
    exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_identity=0.5, continue_train=False,
              extra=f" --choice_spos {choice} --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="cycle_gan", name="cycle_gan_vip_ep400", load_size=256,
                   input_nc=3, output_nc=3, netG='VIP')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, lambda_identity=0.5, continue_train=False,
    #           extra=' --display_ncols 4')  #  --save_latest_freq 200
    # exp.test()  # test with full arch
    # exp.eval(metric_list, testB=False)
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=0, input_nc=1, lambda_identity=0, continue_train=False,
    #           extra=' --DECODE --dataset_mode unaligned')  # retrain
    # exp.test(extra=' --DECODE --dataset_mode unaligned')   # test with full arch
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_mposv1_ep400",
                     load_size=286, netG='mposv1', input_nc=1, output_nc=1,
                     extra=f" --nce_layers 1,2,3,4,5,6")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # NAS
    exp = Experiment(dataset="horse2zebra", model="cut", name="cut_ep400",
                     load_size=286, input_nc=3, output_nc=3,
                     extra=f"")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="horse2zebra", model="cut", name="cut_mposv1_ep400",
                     load_size=286, netG='mposv1', input_nc=3, output_nc=3,
                     extra=f" --nce_layers 1,2,3,4,5,6")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="dints_cycle_gan_mpos", name="dints_cycle_gan_ep10", load_size=256,
                   input_nc=3, output_nc=3)
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, lambda_identity=0.5, continue_train=False,
    #           extra=' --display_ncols 4')  #  --save_latest_freq 200
    # exp.test()  # test with full arch
    # exp.eval(metric_list, testB=False)
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=0, input_nc=1, lambda_identity=0, continue_train=False,
    #           extra=' --DECODE --dataset_mode unaligned')  # retrain
    # exp.test(extra=' --DECODE --dataset_mode unaligned')   # test with full arch
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    #
    exp = Experiment(dataset="horse2zebra", model="cut", name="cut_hg_ep400", load_size=256, netG='hourglass',
                     input_nc=3, output_nc=3,
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=200, nce_idt=True, continue_train=True,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="unit", name="unit_ep400", load_size=256,
                   input_nc=3, output_nc=3)
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="unitv2", name="unitv2_ep200", load_size=256,
                   input_nc=3, output_nc=3)
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # show_results(metric_list, rowsA, rowsB)


def dog2cat():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    # NAS
    exp = Experiment(dataset="horse2zebra", model="dints_cut_mpos", name="dints_cut_mpos_ep400", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, extra=f" --nce_layers 0,1,2,3,4")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, nce_idt=True, continue_train=False,
    #           display_ncols=4, extra='')
    # exp.test()
    # exp.eval(metric_list, testB=False)
    #
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=50, nce_idt=True, continue_train=True, display_ncols=4,
    #           extra=' --DECODE')
    # exp.test(extra=' --DECODE')
    exp.fid()
    # exp.eval(metric_list, testB=False)

    #
    exp = Experiment(dataset="horse2zebra", model="cut", name="cut_hg_ep400", load_size=256, netG='hourglass',
                     input_nc=3, output_nc=3,
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=200, nce_idt=True, continue_train=True,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="unit", name="unit_ep400", load_size=256,
                   input_nc=3, output_nc=3)
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="unitv2", name="unitv2_ep200", load_size=256,
                   input_nc=3, output_nc=3)
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def hyper_pm():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # 1280 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dvfreq", name="cut3dvfreq_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # # ##############################  IXI  ############################### # #
    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep20_widt", load_size=256)
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1690 sec/epoch
    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10_widt", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10_bs4_lr4", load_size=256)
    # exp.train(batch_size=4, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10_bs6_lr5", load_size=256)
    # exp.train(batch_size=6, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep15_bs6_lr5", load_size=256)
    # exp.train(batch_size=6, n_epochs=5, n_epochs_decay=10, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutpp", name="cutpp_ep10", load_size=256,
                     dataset_mode="unaligned4A2B1")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="fccut", name="fccut_ep10", load_size=256,
                     dataset_mode="unaligned4FCCUT")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=True,
    #           extra=" --reg_start_epoch 0", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1718 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep10_widt", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # 1718 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep20_widt", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep10_bs4", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=4, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra=" --lr ", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep10_bs4_lr4", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=4, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra=" --lr 0.0004", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep15_bs4_lr4", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=4, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra=" --lr 0.0004", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep10_bs6", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=6, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut3damp", name="cut3damp_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 700 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3damp", name="cut3damp_ep10_bs6", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=6, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB, testB=False)


def greedy_search():
    metric_list = ['psnr', 'ssim']

    choices = []
    n_choices = 5
    for a in range(n_choices):
        for b in range(n_choices):
            for c in range(n_choices):
                for d in range(n_choices):
                    choices.append([a, b, c, d])
                    # for e in range(n_choices):
                    #     choices.append([a, b, c, d, e])
    choices = [[0, 2, 3, 3], [1, 2, 3, 3], [2, 2, 3, 3]]
    psnr, ssim = [], []

    import random
    random.shuffle(choices)
    # choices = choices[:50]

    for choice in choices:
        exp = CycleGAN(dataset="IXI", model="cycle_ganspos", name="cyclegan_spos_ep20", load_size=256, netG='spos',
                       extra=" --ngf 64 --n_blocks 4 --n_choices 4")
        # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=15, input_nc=1, lambda_identity=0, continue_train=False)
        prefix = str(choice).replace(' ', '')
        exp.test(script='test_spos', extra=f" --choice_spos {prefix}")
        # exp.fid()
        ret, _ = exp.eval_spos(metric_list, choice_spos=prefix, testB=False)
        psnr.append(ret[0])
        ssim.append(ret[1])

    # psnr_max_idx = sorted(range(len(psnr)), key=lambda i: psnr[i])
    # ssim_max_idx = sorted(range(len(ssim)), key=lambda i: ssim[i])
    # print("PSNR", ssim)
    # print(psnr, ssim)

    import pickle
    with open("eval_spos.txt", "wb") as f:
        pickle.dump({"choices": choices, "psnr": psnr, "ssim": ssim}, f)
    print(choices[10:])
    print(psnr, ssim)
    show_results(metric_list, rowsA, rowsB, testB=False)


def greedy_search_cut():
    metric_list = ['psnr', 'ssim']

    choices = []
    n_choices = 4
    n_blocks = 6
    choices = list(itertools.combinations_with_replacement(range(n_choices), n_blocks))
    # choices = [[0, 0, 0, 0, 0, 0], [3, 3, 2, 2, 2, 3], [3, 0, 2, 2, 0, 3]]
    psnr, ssim = [], []

    import random
    random.shuffle(choices)
    choices = choices[:30]

    for choice in choices:
        exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep30", load_size=256, netG='spos',
                         extra=" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14")
        # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=15, input_nc=1, lambda_identity=0, continue_train=False)
        prefix = str(choice).replace(' ', '')
        exp.test(script='test_spos', extra=f" --choice_spos {prefix}")
        # exp.fid()
        ret, _ = exp.eval_spos(metric_list, choice_spos=prefix, testB=False)
        psnr.append(ret[0])
        ssim.append(ret[1])

    # psnr_max_idx = sorted(range(len(psnr)), key=lambda i: psnr[i])
    # ssim_max_idx = sorted(range(len(ssim)), key=lambda i: ssim[i])
    # print("PSNR", ssim)
    # print(psnr, ssim)

    import pickle
    with open("eval_spos.txt", "wb") as f:
        pickle.dump({"choices": choices, "psnr": psnr, "ssim": ssim}, f)
    print(choices[10:])
    print(psnr, ssim)
    show_results(metric_list, rowsA, rowsB, testB=False)


def our_proposal():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    # dcl, 54s, bs=2
    # exp = Experiment(dataset="ct2mr", name="dcl")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=400, continue_train=True)
    # exp.test()
    # exp.eval(ssim)

    # # ##############################  IXI  ############################### # #

    exp = Experiment(dataset="IXI", model="cut2f", name="cut2f_hrt_ep20", load_size=256, netG='hrt_nb',
                     extra=' --n_blocks 6 --nce_layers 0,4,8,12,13,14', input_nc=1, output_nc=1)
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=15, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Pix2Pix(dataset="IXI", model="pix2pix", name="pix2pix_unet256_ep10", load_size=256, netG="unet_256",
                  input_nc=1, output_nc=1, dataset_mode='alignedslices',
                  extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # # 1840 sec/epoch
    # exp = Pix2Pix(dataset="IXI", model="pix2pix", name="pix2pix_resnet_ep10", load_size=256, netG="resnet_9blocks",
    #               input_nc=1, output_nc=1, dataset_mode='alignedslices',
    #               extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # # exp.test()
    # # # # # exp.fid()
    # # exp.eval(metric_list, testB=False)
    #
    # # 1840 sec/epoch
    # exp = Pix2Pix(dataset="IXI", model="pix2pix", name="pix2pix_hg_ep10", load_size=256, netG="hourglass",
    #               input_nc=1, output_nc=1, dataset_mode='alignedslices',
    #               extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Pix2Pix(dataset="IXI", model="pix2pixnd", name="pix2pix3d_unet256_ep10", load_size=256, netG="unet_256",
                  input_nc=1, output_nc=1, dataset_mode='alignedslices',
                  extra=" --num_K 1 --ngf 64")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Pix2Pix(dataset="IXI", model="pix2pixnd", name="pix2pix3d_unet256_ep10", load_size=256, netG="resnet_9blocks",
                  input_nc=1, output_nc=1, dataset_mode='alignedslices',
                  extra=" --num_K 1 --ngf 64")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # # exp.test()
    # # # # # exp.fid()
    # # exp.eval(metric_list, testB=False)
    #
    # # 1840 sec/epoch
    # exp = Pix2Pix(dataset="IXI", model="pix2pixnd", name="pix2pix3d_unet256_ep10", load_size=256, netG="hourglass",
    #               input_nc=1, output_nc=1, dataset_mode='alignedslices',
    #               extra=" --num_K 1 --ngf 64")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_L1=100,
    #           extra=" --save_latest_freq 5000 --display_ncols=3")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutndfc", name="cut5dfc_hge+1_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --lambda_FC 10   --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutndfc", name="cut5dfc_ep10", load_size=256,
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --lambda_FC 0.01 --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutndfc", name="cut5dfc_hge-2_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --lambda_FC 0.01 --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_resvit_ep10", load_size=256, netG='resvit',
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutndfc", name="cut5dfc_hge-1_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --lambda_FC 0.1 --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutndfc", name="cut5dfc_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --lambda_FC 1 --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd", name="cutndk3_ep10", load_size=256,
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 1 --ngf 64")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd", name="cutndk5_ep10", load_size=256,
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd", name="cutndk7_ep10", load_size=256,
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1840 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd", name="cut5d_hg_ngf96_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 96 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1206 sec/epoch
    exp = Experiment(dataset="IXI", model="cutnd", name="cut5d_hgk3_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 5 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutnd", name="cut5d_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 2 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutnd", name="cut7d_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 3 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_ncols=4")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutnd", name="cutnd_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices',
                     extra=" --num_K 5 --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unit_hg", name="unit_hg_ep10", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unit_hg", name="unit_hg_ep10_wod", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unitv2_hg", name="unitv2_hg_ep10", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unitv2", name="unitv2_ep10", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unit", name="unit_ep20", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unit", name="unit_ep10", load_size=256,
                   input_nc=1, output_nc=1)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, continue_train=False, lambda_identity=0,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutdcl", name="cutdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="unit3d", name="unit3d_ep10", load_size=256,
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut3dv2freq", name="cut3dv2freq_hg_ep10", load_size=256, netG='hourglass',
                     dataset_mode="unaligned4seq2seq", extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_inred_ep10", load_size=256, netG='inred',
                   extra=" --ngf 16")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_ganspos", name="cyclegan_spos_ep20", load_size=256, netG='spos',
                   extra=" --ngf 48 --n_blocks 4 --n_choices 4")
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cuthourglass_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_Gfreq_ep10", load_size=256, netG='frequency',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_Gfreq_ep10", load_size=256, netG='frequency',
                     input_nc=1, output_nc=1, dataset_mode="unaligned4seq2seq",
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000", script_name="train4fccut")
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cycle_gan_Gfreq_ep10", load_size=256, netG='frequency')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cycle_gan3d_Gfreq_ep10", load_size=256, netG='frequency',
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    ## CycleGAN

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cycle_gan_hg_ep10", load_size=256, netG='hourglass')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cycle_gan3d_hg_ep10", load_size=256, netG='hourglass',
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hg_ep10_widt", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000", script_name="train4fccut")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1,
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl3d", name="dcl3d_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode="unaligned4seq2seq",
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000", script_name="train4fccut")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut3d", name="cut3dhourglass_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode="unaligned4seq2seq",
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
    #           extra=" --save_latest_freq 5000", script_name="train4fccut")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut3d", name="cut3dhourglass_ep10_widt", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode="unaligned4seq2seq",
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True   , continue_train=False,
    #           extra=" --save_latest_freq 5000", script_name="train4fccut")
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep30", load_size=256, netG='spos',
                     input_nc=3, output_nc=3,
                     extra=" --ngf 64 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,13,14")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=120, lambda_identity=0, continue_train=True, extra=" --save_latest_freq 5000")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep30", load_size=256, netG='spos',
                     extra=" --ngf 48 --n_blocks 5 --n_choices 4 --nce_layers 0,4,8,12,14 --save_latest_freq 5000")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=True)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    choice = str([0, 1, 0, 0]).replace(' ', '')
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_0100_ep10", load_size=256, netG='spos',
                     extra=f" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, input_nc=1, lambda_identity=0, continue_train=True)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    choice = str([0, 0, 1, 1, 2, 2]).replace(' ', '')
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_001122_ep30_wotd", load_size=256, netG='spos',
                     extra=f" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=30, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    choice = str([0, 0, 1, 1, 2, 2]).replace(' ', '')
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_001122_ep10", load_size=256, netG='spos',
                     extra=f" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=True)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    choice = str([0, 0, 1, 1, 2, 2]).replace(' ', '')
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_001122_ep10_wot", load_size=256, netG='spos',
                     extra=f" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    choice = str([0, 0, 1, 1, 2, 2]).replace(' ', '')
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_001122_ep10_wotd", load_size=256, netG='spos',
                     extra=f" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    choice = str([0, 0, 1, 1, 2, 2]).replace(' ', '')
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_001122_ep20_wotd", load_size=256, netG='spos',
                     extra=f" --ngf 48 --n_blocks 6 --n_choices 4 --nce_layers 0,4,8,12,14 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=20, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut3dv1", name="cut3dv1_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=2, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=True,
    #           extra=" --no_flip", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cyclegan3d_hrt_ep10", load_size=256, netG='hrt',
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan3d", name="cyclegan3d_ep10", load_size=256,
                   dataset_mode="unaligned4seq2seq")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_hrt_ep10", load_size=256, netG='hrt_nb',
                     extra=' --n_blocks 3')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="hrcut", name="hrcut_ep10", load_size=256,
                     netG='hrt_nb', extra=' --n_blocks 3 --nce_layers 0,4,8,12')
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, input_nc=1, nce_idt=True, continue_train=True)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_hrt_ep10", load_size=256,
                     netG='hrt_nb', extra=' --n_blocks 3 --nce_layers 0,4,8,12')
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, input_nc=1, continue_train=False,
    #           extra=" --lambda_NCE 0")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_hrt4_ep10", load_size=256,
                     netG='hrt_nb', extra=' --n_blocks 4',
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_hrt_ep10", load_size=256,
                     netG='hrt_nb', extra=' --n_blocks 3',
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_hrt_ep10_widt", load_size=256,
                     netG='hrt_nb', extra=' --n_blocks 3',
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_hrt_ep10", load_size=256, netG='hrt')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="BraTS18", model="cycle_gan", name="cyclegan_hrt_ep10", load_size=256, netG='hrt')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_hrt2_ep10", load_size=256, netG='hrt_2blocks')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_hrt_ep10_widt", load_size=256, netG='hrt')
    # exp.train(batch_size=1, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0.5, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_hrtidt_ep10", load_size=256,
                   netG='hrt', extra=' --skip_connec')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_hrtidt_ep10_widt", load_size=256,
                   netG='hrt', extra=' --skip_connec')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0.5, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_lptn_ep10", load_size=256, netG='lptn')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="LIVECell", model="dcl4is", name="dcl4is_ep200", load_size=256,
                     input_nc=3, output_nc=3,
                     dataset_mode="unaligned4is")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, input_nc=3, output_nc=3, nce_idt=True,
    #           continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1104 sec/epoch
    exp = Experiment(dataset="IXI", model="cutpp", name="cutpp_ep10", load_size=256,
                     dataset_mode="unaligned4A2B2")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=" --display_ncols 4")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="fccut", name="fccut_ep10", load_size=256,
                     dataset_mode="unaligned4FCCUT")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=" --reg_start_epoch 5 --no_flip", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="fccutv3", name="fccutv3_ep10", load_size=256,
                     dataset_mode="unaligned4FCCUT")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False,
    #           extra=" --reg_start_epoch 15 --no_flip", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 2820 sec/epoch
    exp = Experiment(dataset="IXI", model="dcl3d", name="dcl3d_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3d", name="cut3d_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dv4", name="cut3dv4_ep10", load_size=256,
                     dataset_mode="unaligned4stylence")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=4,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1280 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dv1", name="cut3dv1_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1280 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dvfreq", name="cut3dvfreq_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1480 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dv2freq", name="cut3dv2freq_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 2095 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dv2freq", name="cut3dv2freq_ep10_widt", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1480 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dv3freq", name="cut3dv3freq_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=False, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1085 sec/epoch
    exp = Experiment(dataset="IXI", model="cut3dv2", name="cut3dv2_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #          nce_idt=True, continue_train=False,
    #          extra=" --lambda_FC 10", display_ncols=3,
    #          script_name='train4fccut')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB, testB=False)


def our_proposal_brats18():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    # # ##############################  IXI  ############################### # #
    ## base + 3D
    # 1598 sec/epoch
    exp = CycleGAN(dataset="BraTS18", model="cycle_gan3d", name="cycle_gan3d_ep10", load_size=256,
                   dataset_mode="unaligned4seq2seq")
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # 1000 sec/epoch
    exp = Experiment(dataset="BraTS18", model="cut3d", name="cut3d_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
              nce_idt=False, continue_train=False,
              extra="", display_ncols=3,
              script_name='train4fccut')
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="BraTS18", model="dcl3d", name="dcl3d_ep10", load_size=256,
                     dataset_mode="unaligned4seq2seq")
    # input_nc is 3, actually
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, output_nc=1,
    #           nce_idt=True, continue_train=False,
    #           extra="", display_ncols=3,
    #           script_name='train4fccut')
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    ## HG + 3D
    # 1631 sec/epoch
    exp = CycleGAN(dataset="BraTS18", model="cycle_gan3d", name="cycle_gan3d_hg_ep10", load_size=256, netG='hourglass',
                   dataset_mode="unaligned4seq2seq")
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    # 1008 sec/epoch
    exp = Experiment(dataset="BraTS18", model="cut3d", name="cut3d_hg_ep10", load_size=256, netG='hourglass',
                     input_nc=1, output_nc=1, dataset_mode="unaligned4seq2seq",
                     extra=" --ngf 64 --nce_layers 0,4,8,9,13")
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False,
              extra=" --save_latest_freq 5000", script_name="train4fccut")
    exp.test()
    # exp.fid()
    exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB, testB=False)


def our_proposal_SR():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    metric_list += ['total_variation', 'brisque']

    # # ##############################  IXI  ############################### # #
    exp = Experiment(dataset="IXI-SR", model="cutsr", name="cutsr_ep20", load_size=256, dataset_mode='unalignedSR')
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=20, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI-SR", model="cutreg", name="cutreg_ep20", load_size=256, dataset_mode='unalignedSR')
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=20, input_nc=1, nce_idt=False, continue_train=True)
    # exp.test()
    # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI-SR", model="cutreg", name="cutreg_ep10", load_size=256, dataset_mode='unalignedSR')
    exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    exp.test()
    # # exp.fid()
    exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB, testB=False)

def sifa_mmwhs():
    exp = Experiment(dataset="MMWHS", model="sifav0", name="sifav0_ep4k", load_size=256,
                     netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unaligned4mmwhs', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg")
    exp.train(batch_size=4, n_epochs=25, n_epochs_decay=25, nce_idt=None, continue_train=False,
              extra=" --save_latest_freq 5000 --display_freq 50 --num_classes 5"
                    " --display_ncols 7 --eval_metric --eval_freq 2000")  # --netD basic3d
    # exp.test(phase='train')
    # exp.test(phase='test')
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)


def enco_4_ct2mr():
    exp = Experiment(dataset="MMWHS", model="encov3", name="encov3_ep100", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignednpy', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg")
    exp.train(batch_size=8, n_epochs=50, n_epochs_decay=50, nce_idt=True, continue_train=False,
              extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
                    ' --lambda_IDT 10 --lambda_NCE 10 --netF cam_mlp_sample_nls --stop_idt_epochs 50'
                    ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 100 --gan_mode lsgan  --num_patches 256'
                    ' --prj_norm LN --display_ncols 4')

    exp = Experiment(dataset="original_TRSAA_crop_BY_TUMOR", model="dcl3d", name="dcl3d_ep4k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4seg176x2', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=2, n_epochs=2000, n_epochs_decay=2000, nce_idt=True, continue_train=False,
    #           extra=" --save_latest_freq 5000 --display_freq 50 --serial_batches"
    #                 " --display_ncols 7 --eval_metric --eval_freq 200")  # --netD basic3d
    # exp.test(phase='train')
    # exp.test(phase='test')
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="original_TRSAA_crop_BY_TUMOR", model="encov2", name="encov2_ep8k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4seg176x2', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=8, n_epochs=4000, n_epochs_decay=4000, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 10 --netF cam_mlp_sample_nls --stop_idt_epochs 4000'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 100 --gan_mode lsgan  --num_patches 256'
    #                 ' --prj_norm LN --display_ncols 4')
    # exp.test(phase='train')
    # exp.test(phase='test')
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="original_TRSAA_crop_BY_TUMOR", model="encov1", name="encov1_ep8k", load_size=256, netG='resnet_9blocks',
                     input_nc=1, output_nc=1, dataset_mode='unalignedslices4seg176x2', gpu_ids='0',
                     dataroot="/home/cas/home_ez/Datasets/CT2MR_Reg",
                     extra=" --num_K 0 --ngf 64")
    # exp.train(batch_size=8, n_epochs=4000, n_epochs_decay=4000, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 10 --netF cam_mlp_sample_nls --stop_idt_epochs 4000'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 100 --gan_mode lsgan  --num_patches 256'
    #                 ' --prj_norm LN --display_ncols 4')
    # exp.test(phase='train')
    # exp.test(phase='test')
    # # # exp.fid()
    # exp.eval(metric_list, testB=False)

    # exp.train(batch_size=8, n_epochs=4000, n_epochs_decay=4000, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 10 --netF cam_mlp_sample_nls --stop_idt_epochs 4000'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 100 --gan_mode lsgan  --num_patches 256'
    #                 ' --prj_norm LN --display_ncols 4')

def main():
    # main()
    # baseline_IXI()
    # MR2CT_Reg()
    # BraTS19()
    # enco_4_ct2mr()
    sifa_mmwhs()


if __name__ == '__main__':
    rowsA = []
    rowsB = []

    main()
