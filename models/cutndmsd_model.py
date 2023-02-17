import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss2, StylePatchNCELoss2
import util.util as util
# from .reg_networks import vox_morph_loss
# import voxelmorph as vxm
# import matplotlib.pyplot as plt


class CUTNDMSDModel(BaseModel):
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
        parser.add_argument('--lambda_R', type=float, default=1, help='weight for frame consistency loss')
        parser.add_argument('--lambda_FC', type=float, default=10.0, help='weight for frame consistency loss')
        parser.add_argument('--reg_start_epoch', type=int, default=None, help='when to calculate frame consistency loss')
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

        parser.add_argument('--num_K', type=int, default=10, help='2K + 1 frames')


        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()
        # cut sets nce_idt as True while fastcut don't
        parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.loss_names = ['G_GAN', 'D', 'G', 'NCE']

        if self.isTrain:
            self.visual_names = ['real_A1', 'fake_B1',
                                 'real_A2', 'fake_B2',
                                 'real_A3', 'fake_B3']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B', ]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        opt.input_nc = 2 * self.opt.num_K + 1
        opt.output_nc = 2 * self.opt.num_K + 1
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        # self.netS = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        print(self.netG)

        if self.isTrain:
            opt.output_nc = 1
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            from .reg_networks import VoxelMorph
            from .networks import init_net
            # self.netR = VoxelMorph((opt.input_nc, 256, 256), is_2d=True)
            # from voxelmorph.torch.networks import VxmDense
            # self.netR = VxmDense(inshape=(256, 256))
            # self.netR = init_net(self.netR, opt.init_type, opt.init_gain, opt.gpu_ids)


            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionStyleNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss2(opt).to(self.device))
                self.criterionStyleNCE.append(StylePatchNCELoss2(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=1e-4)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # self.optimizers.append(self.optimizer_R)

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

                # self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                # self.optimizers.append(self.optimizer_S)

            if self.opt.reg_start_epoch is None:
                self.opt.reg_start_epoch = self.opt.n_epochs_decay


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # if self.cur_epoch > self.opt.reg_start_epoch:
        #     # update R
        #     self.optimizer_R.zero_grad()
        #     self.loss_R = self.compute_R_loss()
        #     self.loss_R.backward()
        #     self.optimizer_R.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
            # self.optimizer_S.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            # self.optimizer_S.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.real_A1 = self.real_A[:, [self.real_A.size(1) // 2 - 1]]
        self.real_A2 = self.real_A[:, [self.real_A.size(1) // 2]]
        self.real_A3 = self.real_A[:, [self.real_A.size(1) // 2 + 1]]
        self.real_B1 = self.real_B[:, [self.real_B.size(1) // 2 - 1]]
        self.real_B2 = self.real_B[:, [self.real_B.size(1) // 2]]
        self.real_B3 = self.real_B[:, [self.real_B.size(1) // 2 + 1]]
        #
        # if self.opt.isTrain:
        #     self.real_R1 = input['R_pre'].to(self.device)
        #     self.real_R2 = input['R'].to(self.device)
        #     self.real_R3 = input['R_post'].to(self.device)
        #
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
            self.idt_B = self.fake[self.real_A.size(0):, [self.real_A.size(1) // 2]]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # fake = self.fake_B.detach()
        # self.loss_D = self.calculate_StyleNCE_loss(self.real_A, fake)
        # self.loss_D_real = 0
        # self.loss_D_fake = 0

        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_7, pred_1 = self.netD(torch.cat([fake.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size),
                                         self.real_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size)], dim=0))
        self.loss_D_fake = self.criterionGAN(pred_7[:self.fake.shape[0]], False).mean() \
                           + self.criterionGAN(pred_1[:self.fake.shape[0]], False).mean()
        # Real
        self.loss_D_real = self.criterionGAN(pred_7[self.fake.shape[0]:], True).mean() \
                           + self.criterionGAN(pred_1[self.fake.shape[0]:], True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
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
            pred_fake_7, pred_fake_1 = self.netD(fake)
            self.loss_G_GAN = (self.criterionGAN(pred_fake_7, True).mean() + self.criterionGAN(pred_fake_1, True).mean()) * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # maintain content
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        # identity loss
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
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

    def calculate_StyleNCE_loss(self, src, tgt):
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
        feat_v = self.netG(self.real_B, self.nce_layers, encode_only=True)

        # MLP
        # [torch.Size([512, 256]), ...] [torch.Size([256]), ...]
        feat_k_pool, sample_ids = self.netS(feat_k, self.opt.num_patches, None)
        # [torch.Size([512, 256]), ...]
        feat_q_pool, _ = self.netS(feat_q, self.opt.num_patches, sample_ids)
        feat_v_pool, _ = self.netS(feat_v, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, f_v, crit, nce_layer in zip(feat_q_pool, feat_k_pool, feat_v_pool, self.criterionStyleNCE,
                                                  self.nce_layers):
            # (B, Dim) ,e.g., torch.Size([512, 256])
            loss = crit(f_q, f_k, f_v) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

