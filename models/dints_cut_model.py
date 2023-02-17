import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss2
import util.util as util
import os


class DINTSCUTModel(BaseModel):
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
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()     
        parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser


    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        from models.blocks.dints_monai import TopologySearch, TopologyInstance, DiNTS

        spatial_dims = 2
        channel_mul = 0.75  # 0.5
        num_depths = 3  # 4
        num_blocks = 6  # 12
        num_ops = 3

        # train / test with full arch
        if not self.opt.DECODE:
            self.dints_space = TopologySearch(
                channel_mul=channel_mul,
                num_blocks=num_blocks,
                num_depths=num_depths,
                use_downsample=True,
                spatial_dims=spatial_dims,
                device='cuda:0',
            )

            self.netG = DiNTS(
                dints_space=self.dints_space_A,
                spatial_dims=spatial_dims,
                in_channels=opt.input_nc,
                num_classes=1,
                use_downsample=True,
            ).cuda()

        else:
            ckptA = torch.load(os.path.join(self.opt.checkpoints_dir, self.opt.name, "search_code_A" + ".pth"))
            node_a = ckptA['node_a']
            arch_code_a = ckptA['arch_code_a']
            arch_code_c = ckptA['arch_code_c']

            import numpy as np
            print(node_a, np.array(node_a).shape)            # 10 x 3
            print(arch_code_a, np.array(arch_code_a).shape)  # 9  x 7
            print(arch_code_c, np.array(arch_code_c).shape)  # 9  x 7 i.e., blocks x edges

            print('Decoding arch of netG_A...')

            self.dints_space = TopologyInstance(
                channel_mul=channel_mul,
                num_blocks=num_blocks,
                num_depths=num_depths,
                use_downsample=True,
                spatial_dims=spatial_dims,
                arch_code=[arch_code_a, arch_code_c],
                device='cuda:0',
            )

            self.netG = DiNTS(
                dints_space=self.dints_space_A,
                spatial_dims=spatial_dims,
                in_channels=opt.input_nc,
                num_classes=opt.output_nc,
                node_a=node_a,
                use_downsample=True,
            ).cuda()

        # define networks (both generator and discriminator)
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers * num_depths:
                self.criterionNCE.append(PatchNCELoss2(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.weight_parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if not self.opt.DECODE:
                self.optimizer_G_arch_a = torch.optim.Adam([self.dints_space.log_alpha_a], lr=opt.lr_arch,
                                                            betas=(opt.beta1, 0.999), weight_decay=0.0)
                self.optimizer_G_arch_c = torch.optim.Adam([self.dints_space.log_alpha_c], lr=opt.lr_arch,
                                                            betas=(opt.beta1, 0.999), weight_decay=0.0)
                # self.optimizers.append(self.optimizer_G_arch_a)
                # self.optimizers.append(self.optimizer_G_arch_c)

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
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        if not self.opt.DECODE:
            # # update weight of model parameters
            self.set_requires_grad(self.netG, True)
            self.dints_space.log_alpha_a.requires_grad = False
            self.dints_space.log_alpha_c.requires_grad = False

        if not self.opt.DECODE:
            # # update weight of model arch
            self.set_requires_grad(self.netG, False)
            self.dints_space.log_alpha_a.requires_grad = True
            self.dints_space.log_alpha_c.requires_grad = True

            # forward_arch
            self.forward_arch()  # compute fake images and reconstruction images.

            self.set_requires_grad(self.netD, False)
            self.optimizer_G_arch_a.zero_grad()
            self.optimizer_G_arch_c.zero_grad()
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.zero_grad()
            self.loss_G_ar = self.compute_G_loss_arch()
            self.loss_G_ar.backward()

            self.optimizer_G_arch_a.step()
            self.optimizer_G_arch_c.step()
            if self.opt.netF == 'mlp_sample':
                self.optimizer_F.step()

        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if not self.opt.DECODE:
            self.real_A_ar = input['Aa' if AtoB else 'Ba'].to(self.device)
            self.real_B_ar = input['Ba' if AtoB else 'Aa'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def forward_arch(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_ar = torch.cat((self.real_A_ar, self.real_B_ar), dim=0) \
            if self.opt.nce_idt and self.opt.isTrain else self.real_A_ar
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real_ar = torch.flip(self.real_ar, [3])

        self.fake_ar = self.netG(self.real_ar)
        self.fake_B_ar = self.fake_ar[:self.real_A_ar.size(0)]
        if self.opt.nce_idt:
            self.idt_B_ar = self.fake[self.real_A_ar.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_D_loss_arch(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B_ar.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake_ar = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real_ar = self.netD(self.real_B_ar)
        loss_D_real = self.criterionGAN(self.pred_real_ar, True)
        self.loss_D_real_ar = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D_ar = (self.loss_D_fake_ar + self.loss_D_real_ar) * 0.5
        return self.loss_D_ar

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def compute_G_loss_arch(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B_ar
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN_ar = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN_ar = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE_ar = self.calculate_NCE_loss(self.real_A_ar, self.fake_B_ar)
        else:
            self.loss_NCE_ar, self.loss_NCE_bd_ar = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y_ar = self.calculate_NCE_loss(self.real_B_ar, self.idt_B_ar)
            loss_NCE_both = (self.loss_NCE_ar + self.loss_NCE_Y_ar) * 0.5
        else:
            loss_NCE_both = self.loss_NCE_ar

        self.loss_G_ar = self.loss_G_GAN_ar + loss_NCE_both
        return self.loss_G_ar

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
