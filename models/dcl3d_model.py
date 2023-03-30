import itertools
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from util.image_pool import ImagePool

from monai.metrics import DiceMetric, MeanIoU
from monai.utils.enums import MetricReduction
from .RegGAN import Reg
from .networks import init_net, Upsample2

from util.util import AvgrageMeter
import numpy as np
import os
from functools import partial

import SimpleITK as sitk
from monai.inferers import sliding_window_inference


class DCL3DModel(BaseModel):
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for DCLGAN """
        parser.add_argument('--DCL_mode', type=str, default="DCL", choices='DCL')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=2.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=1.0, help='weight for l1 identical loss: (G(X),X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization.")

        parser.add_argument('--num_K', type=int, default=10, help='2K + 1 frames')
        parser.add_argument('--lambda_R', type=float, default=0.5, help='weight for frame consistency loss')

        parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
        parser.add_argument('--lazy_reg', type=int, default=None,
                            help='lazy regulariation.')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for DCLGAN.
        if opt.DCL_mode.lower() == "dcl":
            parser.set_defaults(nce_idt=True, lambda_NCE=2.0)
        else:
            raise ValueError(opt.DCL_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G', 'Seg', 'Reg']
        visual_names_A = ['real_A', 'real_A_SEG', 'fake_B', 'fake_B_SEG', 'fake_REG_B', 'real_RB']
        visual_names_B = ['real_B', 'real_B_SEG', 'fake_A', 'fake_A_SEG', 'fake_REG_A', 'real_RA']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        visual_names_B.append('real_B_gt')
        self.loss_names += ['Dice_train', 'IoU_train', 'Dice_val', 'IoU_val']

        self.loss_Dice_train = 0
        self.loss_IoU_train = 0
        self.loss_Dice_val = 0
        self.loss_IoU_val = 0

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B', 'S_A', 'S_B', 'R_A', 'R_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netF1 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        opt.input_nc = 1
        self.netS_A = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, 'unet_256', opt.normG, not opt.no_dropout,
                                        opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                        self.gpu_ids, opt)
        self.netS_B = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, 'unet_256', opt.normG, not opt.no_dropout,
                                        opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up,
                                        self.gpu_ids, opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netR_A = Reg(in_channel=2 * self.opt.num_K + 1).cuda()
            self.netR_A = init_net(self.netR_A, opt.init_type, opt.init_gain, opt.gpu_ids)

            self.netR_B = Reg(in_channel=2 * self.opt.num_K + 1).cuda()
            self.netR_B = init_net(self.netR_B, opt.init_type, opt.init_gain, opt.gpu_ids)

            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSEG = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.])).to(self.device)

            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_S = torch.optim.Adam(itertools.chain(self.netS_A.parameters(), self.netS_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(itertools.chain(self.netR_A.parameters(), self.netR_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_S)
            self.optimizers.append(self.optimizer_R)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()  # calculate graidents for G
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        self.optimizer_R.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()
        self.optimizer_R.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input, test=False):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        assert self.opt.direction == 'AtoB', 'only support AtoB.'
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_RA = input['RA' if AtoB else 'RB'].to(self.device)
        self.real_RB = input['RB' if AtoB else 'RA'].to(self.device)

        self.real_B_gt = input['B_gt'].to(self.device)
        self.real_B_gt[self.real_B_gt > 0] = 1
        self.real_B_gt[self.real_B_gt <= 0] = 0
        self.real_B_gt = self.real_B_gt.squeeze(1).permute(0, 3, 2, 1)
        if test:
            self.real_A_gt = input['A_gt'].to(self.device)
            self.real_A_gt[self.real_A_gt > 0] = 1
            self.real_A_gt[self.real_A_gt <= 0] = 0
            self.real_A_gt = self.real_A_gt.squeeze(1).permute(0, 3, 2, 1)

        if self.opt.isTrain:
            self.real_A = self.real_A.squeeze(1).permute(0, 3, 2, 1)
            self.real_B = self.real_B.squeeze(1).permute(0, 3, 2, 1)
            self.real_RA = self.real_RA.squeeze(1).permute(0, 3, 2, 1)
            self.real_RB = self.real_RB.squeeze(1).permute(0, 3, 2, 1)
        else:
            self.real_A_gt = input['A_gt'].to(self.device)
            self.real_B_gt = input['B_gt'].to(self.device)

            self.image_meta = input['A_meta' if AtoB else 'B_meta']

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        AtoB = self.opt.phase == 'test'

        model = self.netG_A if AtoB else self.netG_B
        real_A = self.real_A if AtoB else self.real_B
        real_B = self.real_B if AtoB else self.real_A
        real_A_gt = self.real_A_gt if AtoB else self.real_B_gt

        self.opt.patch_size = (256, 256, 1)
        model_AtoB = partial(sliding_window_inference,
                             roi_size=(self.opt.patch_size[0], self.opt.patch_size[1], self.opt.patch_size[2]),
                             sw_batch_size=1, predictor=self.netG_A, overlap=0.5)
        model_BtoA = partial(sliding_window_inference,
                             roi_size=(self.opt.patch_size[0], self.opt.patch_size[1], self.opt.patch_size[2]),
                             sw_batch_size=1, predictor=self.netG_B, overlap=0.5)

        with torch.no_grad():
            print(real_A.shape)
            fake_B = model_AtoB(real_A)
            fake_A = model_BtoA(real_B)
            # print(fake_B.shape, real_A_gt.shape)

            fake_A_list = [torch.rot90(fake_A.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]
            real_B_list = [torch.rot90(real_B.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]
            fake_B_list = [torch.rot90(fake_B.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]

            real_A_gt_list = [torch.rot90(real_A_gt.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]

        img_paths = self.get_image_paths()

        save_dir = os.path.join(self.opt.results_dir, self.opt.name, f'{self.opt.phase}_{self.opt.epoch}', 'images')

        fake_A_path = 'testA' if AtoB else 'trainA'
        real_A_gt_path = 'testA_gt' if AtoB else 'trainA_gt'
        fake_B_path = 'testB' if AtoB else 'trainB'

        os.makedirs(os.path.join(save_dir, fake_A_path), exist_ok=True)
        os.makedirs(os.path.join(save_dir, real_A_gt_path), exist_ok=True)
        os.makedirs(os.path.join(save_dir, fake_B_path), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'real_B'), exist_ok=True)

        for i in range(len(img_paths)):
            path = img_paths[i]
            name = os.path.basename(path)

            real = torch.nn.functional.interpolate(fake_A_list[i], (176, 176), mode='bilinear')[0]  # * 1000
            real_B = torch.nn.functional.interpolate(real_B_list[i], (176, 176), mode='bilinear')[0]  # * 1000
            fake = torch.nn.functional.interpolate(fake_B_list[i], (176, 176), mode='bilinear')[0]  # * 1000
            real = (real - real.min()) / (real.max() - real.min())
            real_B = (real_B - real_B.min()) / (real_B.max() - real_B.min())
            fake = (fake - fake.min()) / (fake.max() - fake.min())

            label_a = real_A_gt_list[i][0]
            # fake = (fake + 1) / 2 * 1000
            # real = (real + 1) / 2 * 1000

            nii = sitk.GetImageFromArray(fake.cpu().numpy())
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, fake_B_path, name))

            nii = sitk.GetImageFromArray(real.cpu().numpy())
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, fake_A_path, name))

            nii = sitk.GetImageFromArray(real_B.cpu().numpy())
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, 'real_B', name))

            nii = sitk.GetImageFromArray(label_a.cpu().numpy().astype('int'))
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, real_A_gt_path, name))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.opt.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

        # segmentation
        self.fake_A_SEG = self.netS_A(self.fake_A)
        self.real_A_SEG = self.netS_A(self.real_A)

        self.fake_B_SEG = self.netS_B(self.fake_B)
        self.real_B_SEG = self.netS_B(self.real_B)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_GAN

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_GAN

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        # maintain content
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.opt.lambda_NCE
            self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0

        if self.opt.lambda_NCE > 0.0:
            # L1 IDENTICAL Loss
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_IDT
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_IDT
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5
        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5

        # reg loss
        self.fake_REG_A, dvf = self.netR_A(self.fake_A, self.real_RA)
        self.loss_Reg_A = self.opt.lambda_R * self.calculate_reg_loss((self.fake_REG_A, dvf), self.real_RA)

        self.fake_REG_B, dvf = self.netR_B(self.fake_B, self.real_RB)
        self.loss_Reg_B = self.opt.lambda_R * self.calculate_reg_loss((self.fake_REG_B, dvf), self.real_RB)

        self.loss_Reg = 0.5 * (self.loss_Reg_A + self.loss_Reg_B)

        # segmentation loss
        self.loss_Seg = self.criterionSEG(torch.cat([self.real_B_SEG, self.fake_A_SEG], dim=0),
                                          torch.cat([self.real_B_gt, self.real_B_gt], dim=0)) * 5

        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 + loss_NCE_both + self.loss_Seg + self.loss_Reg
        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def calculate_reg_loss(self, src, tgt):
        # print(tgt.shape, src[0].shape, src[1].shape)
        # prepare image loss
        loss_image = self.criterionIdt(tgt, src[0]) * 20

        # prepare deformation loss
        loss_deform = self.smooothing_loss(src[1]) * 10

        # print(loss_image.item(), loss_deform.item())
        return loss_image + loss_deform

    def smooothing_loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dx = dx * dx
        dy = dy * dy

        return torch.mean(dx) + torch.mean(dy)

    def val_metrics(self, epoch, train_dataloader, val_dataloader):
        self.netS_A.eval()
        self.netS_B.eval()

        dice_log = DiceMetric(include_background=False,
                              reduction=MetricReduction.MEAN_BATCH,
                              get_not_nans=True)
        IoU_log = MeanIoU(include_background=False,
                          reduction=MetricReduction.MEAN_BATCH,
                          get_not_nans=True)
        # train
        dice_train = AvgrageMeter()
        dice_train.reset()
        iou_train = AvgrageMeter()
        iou_train.reset()

        for i, data in enumerate(train_dataloader):
            self.set_input(data)
            # reset
            dice_log.reset()
            IoU_log.reset()

            pred = torch.sigmoid(self.netS_B(self.real_B))
            pred = torch.argmax(torch.cat([1 - pred, pred], dim=1), dim=1, keepdim=True)
            # dice
            dice_log(y_pred=pred, y=self.real_B_gt)
            dice, not_nans = dice_log.aggregate()
            dice_train.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

            # iou
            IoU_log(y_pred=pred, y=self.real_B_gt)
            iou, not_nans = IoU_log.aggregate()
            iou_train.update(iou.cpu().numpy(), n=not_nans.cpu().numpy())

        # val
        dice_val = AvgrageMeter()
        dice_val.reset()
        iou_val = AvgrageMeter()
        iou_val.reset()

        for i, data in enumerate(val_dataloader):
            self.set_input(data, test=True)
            # reset
            dice_log.reset()
            IoU_log.reset()

            pred = torch.sigmoid(self.netS_A(self.real_A))
            pred = torch.argmax(torch.cat([1 - pred, pred], dim=1), dim=1, keepdim=True)
            # dice
            dice_log(y_pred=pred, y=self.real_A_gt)
            dice, not_nans = dice_log.aggregate()
            dice_val.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

            # iou
            IoU_log(y_pred=pred, y=self.real_A_gt)
            iou, not_nans = IoU_log.aggregate()
            iou_val.update(iou.cpu().numpy(), n=not_nans.cpu().numpy())

        self.loss_Dice_train = np.nanmean(dice_train.avg)
        self.loss_IoU_train = np.nanmean(iou_train.avg)
        self.loss_Dice_val = np.nanmean(dice_val.avg)
        self.loss_IoU_val = np.nanmean(iou_val.avg)
        print(f"EP: {epoch}, Train Dice: {self.loss_Dice_train}, Train IoU: {self.loss_IoU_train}, "
              f"Val Dice: {self.loss_Dice_val}, Val IoU: {self.loss_IoU_val}")
        self.netS_A.train()
        self.netS_B.train()
