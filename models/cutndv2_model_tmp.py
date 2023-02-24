import numpy as np
import torch
from itertools import chain
from monai.metrics import DiceMetric, MeanIoU
from monai.utils.enums import MetricReduction

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss2, StylePatchNCELoss2
import util.util as util
from .RegGAN import Reg
from .networks import init_net, Upsample2

from util.util import AvgrageMeter


class CUTNDV2Model(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_R', type=float, default=0.5, help='weight for frame consistency loss')
        parser.add_argument('--lambda_FC', type=float, default=10.0, help='weight for frame consistency loss')
        parser.add_argument('--reg_start_epoch', type=int, default=None,
                            help='when to calculate frame consistency loss')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
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
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        # parser.add_argument('--dataset_mode', type=str, default='unalignedslices4reg',)
        parser.add_argument('--num_K', type=int, default=10, help='2K + 1 frames')

        parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
        parser.add_argument('--lazy_reg', type=int, default=None,
                            help='lazy regulariation.')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()
        # cut sets nce_idt as True while fastcut don't
        parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        torch.cuda.set_device(0)
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.loss_names = ['G_GAN', 'D', 'G', 'NCE', 'Reg', 'Seg', 'Seg_Adv', 'D1', 'D2']
        self.loss_Dice_train = 0
        self.loss_IoU_train = 0
        self.loss_Dice_val = 0
        self.loss_IoU_val = 0

        if self.isTrain:
            # self.visual_names = ['real_A1', 'fake_B1',
            #                      'real_A2', 'fake_B2',
            #                      'real_A3', 'fake_B3']
            self.visual_names = ['real_A1', 'fake_B1', 'reg_B1', 'real_R1', 'fake_B1_seg_map', 'real_B1_seg_map',
                                 'real_B1_gt',
                                 'real_A2', 'fake_B2', 'reg_B2', 'real_R2', 'fake_B2_seg_map', 'real_B2_seg_map',
                                 'real_B2_gt',
                                 'real_A3', 'fake_B3', 'reg_B3', 'real_R3', 'fake_B3_seg_map', 'real_B3_seg_map',
                                 'real_B3_gt']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B', ]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            # self.visual_names += ['idt_B']

        self.loss_names += ['Dice_train', 'IoU_train', 'Dice_val', 'IoU_val']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'DX', 'R', 'S']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        opt.input_nc = 2 * self.opt.num_K + 1
        opt.output_nc = 2 * self.opt.num_K + 1
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        opt.input_nc = 1
        self.netS = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, 'unet_256', opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        opt.input_nc = 3
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        print(self.netG)

        if self.isTrain:
            opt.input_nc = 1
            opt.output_nc = 1
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netDX = networks.define_D(opt.output_nc * 2, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                           opt.init_type,
                                           opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netR = Reg(in_channel=2 * self.opt.num_K + 1).cuda()
            self.netR = init_net(self.netR, opt.init_type, opt.init_gain, opt.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSEG = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.])).to(self.device)
            self.criterionNCE = []
            self.criterionStyleNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss2(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(chain(self.netD.parameters(), self.netDX.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=5e-4, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=1e-4)
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
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

            if self.opt.reg_start_epoch is None:
                self.opt.reg_start_epoch = self.opt.n_epochs_decay

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD, self.netDX], True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD, self.netDX], False)
        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()
        self.optimizer_S.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_R.step()
        self.optimizer_S.step()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            # self.optimizer_S.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        assert self.opt.direction == 'AtoB', 'only support AtoB.'
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_R = input['R'].to(self.device)

        self.real_B_gt = input['B_gt'].to(self.device)

        self.real_B_gt[self.real_B_gt > 0] = 1
        self.real_B_gt[self.real_B_gt <= 0] = 0

        self.real_A = self.real_A.squeeze(1).permute(0, 3, 2, 1)
        self.real_B = self.real_B.squeeze(1).permute(0, 3, 2, 1)
        self.real_B_gt = self.real_B_gt.squeeze(1).permute(0, 3, 2, 1)
        self.real_R = self.real_R.squeeze(1).permute(0, 3, 2, 1)

        self.real_B_gt = self.real_B_gt.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)

        # A
        self.real_A1 = self.real_A[:, [self.real_A.size(1) // 2 - 1]]
        self.real_A2 = self.real_A[:, [self.real_A.size(1) // 2]]
        self.real_A3 = self.real_A[:, [self.real_A.size(1) // 2 + 1]]
        # B
        self.real_B1 = self.real_B[:, [self.real_B.size(1) // 2 - 1]]
        self.real_B2 = self.real_B[:, [self.real_B.size(1) // 2]]
        self.real_B3 = self.real_B[:, [self.real_B.size(1) // 2 + 1]]

        if self.opt.isTrain:
            # R
            self.real_R1 = self.real_R[:, [self.real_R.size(1) // 2 - 1]]
            self.real_R2 = self.real_R[:, [self.real_R.size(1) // 2]]
            self.real_R3 = self.real_R[:, [self.real_R.size(1) // 2 + 1]]

        # self.real_A = torch.cat([self.real_A1, self.real_A2, self.real_A3], dim=1)
        # self.real_B = torch.cat([self.real_B1, self.real_B2, self.real_B3], dim=1)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # BCHW -> (2B)CHW
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A

        # if self.opt.flip_equivariance:
        #     self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
        #     if self.flipped_for_equivariance:
        #         self.real = torch.flip(self.real, [3])

        # (2B)CHW -> (2B)CHW
        # print(self.real.shape)
        self.fake = self.netG(self.real)
        # (2B)CHW -> BCHW
        self.fake_B = self.fake[:self.real_A.size(0)]

        if self.isTrain:
            self.fake_B1 = self.fake_B[:, [self.fake_B.size(1) // 2 - 1]]
            self.fake_B2 = self.fake_B[:, [self.fake_B.size(1) // 2]]
            self.fake_B3 = self.fake_B[:, [self.fake_B.size(1) // 2 + 1]]
        else:
            self.real_A = self.real_A[:, [self.real_A.size(1) // 2]]
            self.real_B = self.real_B[:, [self.real_B.size(1) // 2]]
            self.fake_B = self.fake_B[:, [self.fake_B.size(1) // 2]]

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        # Segmentation
        self.real_B_seg_map = self.netS(self.real_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size))
        self.fake_B_seg_map = self.netS(self.fake_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size))

        if self.isTrain:
            self.real_B1_seg_map = torch.sigmoid(self.real_B_seg_map[[self.real_B_seg_map.size(1) // 2 - 1]])
            self.real_B2_seg_map = torch.sigmoid(self.real_B_seg_map[[self.real_B_seg_map.size(1) // 2]])
            self.real_B3_seg_map = torch.sigmoid(self.real_B_seg_map[[self.real_B_seg_map.size(1) // 2 + 1]])

            self.fake_B1_seg_map = torch.sigmoid(self.fake_B_seg_map[[self.fake_B_seg_map.size(1) // 2 - 1]])
            self.fake_B2_seg_map = torch.sigmoid(self.fake_B_seg_map[[self.fake_B_seg_map.size(1) // 2]])
            self.fake_B3_seg_map = torch.sigmoid(self.fake_B_seg_map[[self.fake_B_seg_map.size(1) // 2 + 1]])

            self.real_B1_gt = self.real_B_gt[[self.fake_B_seg_map.size(1) // 2 - 1]]
            self.real_B2_gt = self.real_B_gt[[self.fake_B_seg_map.size(1) // 2]]
            self.real_B3_gt = self.real_B_gt[[self.fake_B_seg_map.size(1) // 2 + 1]]
        else:
            self.real_B_seg_map = self.real_B_seg_map[[self.real_B_seg_map.size(1) // 2]]
            self.fake_B_seg_map = self.fake_B_seg_map[[self.fake_B_seg_map.size(1) // 2]]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        real = self.real_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)
        self.pred_real = self.netD(real)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        fake = self.fake_B.detach().reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real

        self.loss_D1 = (self.loss_D_fake + self.loss_D_real) * 0.5

        # seg adv
        pred_fake = self.netDX(torch.cat([fake,
                                          self.fake_B_seg_map.detach()], dim=1))
        self.loss_DX_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netDX(torch.cat([real,
                                               self.real_B_seg_map.detach()], dim=1))
        self.loss_DX_real = self.criterionGAN(self.pred_real, True).mean()

        self.loss_D2 = (self.loss_DX_fake + self.loss_DX_real) * 0.5

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D1 + self.loss_D2) * 0.5
        return self.loss_D

    # def compute_R_loss(self):
    #     after_R = self.netR(self.before_R, self.real_A)
    #     self.loss_Reg = self.opt.lambda_R * self.criterionReg(after_R, self.real_A)
    #
    #     self.after_R, _ = after_R
    #     # self.after_R = after_R
    #     # self.reg_cmap = plt.cm.coolwarm((self.after_R.detach().squeeze(1) - self.before_R.squeeze(1)).cpu().numpy())[..., :3][[1]].transpose(0, 3, 1, 2)
    #
    #     return self.loss_Reg

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # maintain content
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        # identity loss
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            # self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            self.loss_NCE_Y = self.criterionIdt(self.idt_B, self.real_B) * 10
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # reg loss
        self.fake_reg, dvf = self.netR(self.fake_B, self.real_R)

        self.loss_Reg = self.opt.lambda_R * self.calculate_reg_loss((self.fake_reg, dvf), self.real_R)

        # segmentation loss
        self.loss_Seg = self.criterionSEG(self.real_B_seg_map, self.real_B_gt) * 5

        self.loss_Seg_Adv = self.criterionGAN(
            self.netDX(torch.cat([self.fake_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size),
                                  self.fake_B_seg_map], dim=1)), True).mean()

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_Reg + self.loss_Seg + self.loss_Seg_Adv

        if self.isTrain:
            self.reg_B1 = self.fake_reg[:, [self.fake_B.size(1) // 2 - 1]]
            self.reg_B2 = self.fake_reg[:, [self.fake_B.size(1) // 2]]
            self.reg_B3 = self.fake_reg[:, [self.fake_B.size(1) // 2 + 1]]

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        # get intermediate feature list
        # torch.Size([2, 1, 262, 262])
        # torch.Size([2, 128, 256, 256])
        # torch.Size([2, 256, 128, 128])
        # torch.Size([2, 256, 64, 64])
        # torch.Size([2, 256, 64, 64])
        # query
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        # MLP
        # [torch.Size([512, 256]), ...] [torch.Size([256]), ...]
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        # [torch.Size([512, 256]), ...]
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            # (B, Dim) ,e.g., torch.Size([512, 256])
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

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
        self.netS.eval()

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

            pred = torch.sigmoid(self.netS(self.real_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)))
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
            self.set_input(data)
            # reset
            dice_log.reset()
            IoU_log.reset()

            pred = torch.sigmoid(self.netS(self.real_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)))
            pred = torch.argmax(torch.cat([1 - pred, pred], dim=1), dim=1, keepdim=True)
            # dice
            dice_log(y_pred=pred, y=self.real_B_gt)
            dice, not_nans = dice_log.aggregate()
            dice_val.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

            # iou
            IoU_log(y_pred=pred, y=self.real_B_gt)
            iou, not_nans = IoU_log.aggregate()
            iou_val.update(iou.cpu().numpy(), n=not_nans.cpu().numpy())

        self.loss_Dice_train = dice_train.avg
        self.loss_IoU_train = iou_train.avg
        self.loss_Dice_val = dice_val.avg
        self.loss_IoU_val = iou_val.avg
        print(f"EP: {epoch}, Train Dice: {self.loss_Dice_train}, Train IoU: {self.loss_IoU_train}, "
              f"Val Dice: {self.loss_Dice_val}, Val IoU: {self.loss_IoU_val}")
        self.netS.train()
