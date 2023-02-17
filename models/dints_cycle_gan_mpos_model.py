import random

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
import torch.nn.functional as F
import numpy as np
import pickle


class DINTSCycleGANMPOSModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

            parser.add_argument('--lr_arch', type=float, default=0.001, help='')

        parser.add_argument('--DECODE', action='store_true', help='')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # if not self.opt.DECODE:
        #     self.loss_names += ['dints_A', 'dints_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        from models.blocks.dints_monai_mpos import TopologySearch, TopologyInstance, DiNTS

        self.spatial_dims = 2
        self.channel_mul = 1  # 0.5
        self.num_depths = 3  # 4
        self.num_edges = 3 * self.num_depths - 2
        self.num_blocks = 9  # 12
        self.num_ops = 3

        if not self.opt.DECODE:
            node_a = np.ones((self.num_blocks + 1, self.num_depths))
            arch_code_a = np.ones((self.num_blocks, self.num_edges))
            arch_code_c = np.random.randint(self.num_ops, size=(self.num_blocks, self.num_edges))

            # print(node_a, np.array(node_a).shape)            # 10 x 3
            # print(arch_code_a, np.array(arch_code_a).shape)  # 9  x 7
            # print(arch_code_c, np.array(arch_code_c).shape)  # 9  x 7 i.e., blocks x edges

            print('Decoding arch of netG_A...')

            self.dints_space_A = TopologyInstance(
                channel_mul=self.channel_mul,
                num_blocks=self.num_blocks,
                num_depths=self.num_depths,
                use_downsample=True,
                spatial_dims=self.spatial_dims,
                arch_code=[arch_code_a, arch_code_c],
                device='cuda:0',
            )

            self.netG_A = DiNTS(
                dints_space=self.dints_space_A,
                spatial_dims=self.spatial_dims,
                in_channels=opt.input_nc,
                num_classes=self.opt.output_nc,
                node_a=node_a,
                use_downsample=True,
            ).cuda()

            node_a = np.ones((self.num_blocks + 1, self.num_depths))
            arch_code_a = np.ones((self.num_blocks, self.num_edges))
            arch_code_c = np.random.randint(self.num_ops, size=(self.num_blocks, self.num_edges))

            print('Decoding arch of netG_B...')

            self.dints_space_B = TopologyInstance(
                channel_mul=self.channel_mul,
                num_blocks=self.num_blocks,
                num_depths=self.num_depths,
                use_downsample=True,
                spatial_dims=self.spatial_dims,
                arch_code=[arch_code_a, arch_code_c],
                device='cuda:0',
            )

            self.netG_B = DiNTS(
                dints_space=self.dints_space_B,
                spatial_dims=self.spatial_dims,
                in_channels=opt.input_nc,
                num_classes=self.opt.output_nc,
                node_a=node_a,
                use_downsample=True,
            ).cuda()

            # from .networks import init_net
            # init_net(self.netG_A, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids, initialize_weights=('stylegan2' not in self.netG_A))
            # init_net(self.netG_B, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids, initialize_weights=('stylegan2' not in self.netG_B))

            from util.util import AvgrageMeter

            self.choice = None
            self.mpos_scoreA = {b: {e: {o: AvgrageMeter() for o in range(self.num_ops)} for e in range(self.num_edges)} for b in range(self.num_blocks)}
            self.mpos_scoreB = {b: {e: {o: AvgrageMeter() for o in range(self.num_ops)} for e in range(self.num_edges)} for b in range(self.num_blocks)}

        else:
            ckptB = torch.load(os.path.join(self.opt.checkpoints_dir, self.opt.name, "search_code_B" + ".pth"))
            node_a = ckptB['node_a']
            arch_code_a = ckptB['arch_code_a']
            arch_code_c = ckptB['arch_code_c']

            self.mpos_scoreA = None
            self.mpos_scoreB = None


        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_Aa_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_Ba_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionGAN_arch = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.weight_parameters(),
                                                                self.netG_B.weight_parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if True:
            arch_code_a = torch.ones((self.num_blocks, self.num_edges), device='cuda:0')
            arch_code_c = torch.randint(self.num_ops, size=(self.num_blocks, self.num_edges), device='cuda:0')
            self.archA_code = [arch_code_a, arch_code_c]
            # print(arch_code_c)

            arch_code_a = torch.ones((self.num_blocks, self.num_edges), device='cuda:0')
            arch_code_c = torch.randint(self.num_ops, size=(self.num_blocks, self.num_edges), device='cuda:0')
            self.archB_code = [arch_code_a, arch_code_c]
            # print(arch_code_c)

        self.fake_B = self.netG_A(self.real_A, arch_code=self.archA_code)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B, arch_code=self.archB_code)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B, arch_code=self.archB_code)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A, arch_code=self.archA_code)  # G_A(G_B(B))

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
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B, arch_code=self.archA_code)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A, arch_code=self.archB_code)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

        if self.mpos_scoreA and self.mpos_scoreB:
            for b in range(self.archA_code[1].size(0)):
                for e in range(self.archA_code[1].size(1)):
                    # update A
                    o = self.archA_code[1][b][e].item()
                    reward = (self.loss_G_A + self.loss_cycle_A + self.loss_idt_A).detach().item()
                    self.mpos_scoreA[b][e][o].update(reward)
                    # update B
                    o = self.archB_code[1][b][e].item()
                    reward = (self.loss_G_B + self.loss_cycle_B + self.loss_idt_B).detach().item()
                    self.mpos_scoreB[b][e][o].update(reward)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        # forward
        self.forward()  # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        if random.random() < 1 / self.num_ops:
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

        if not self.opt.DECODE:
            torch.save(
                {
                    "mpos_scoreA": self.mpos_scoreA,
                },
                os.path.join(self.opt.checkpoints_dir, self.opt.name, "mpos_score_A" + ".pth"),
            )

            torch.save(
                {
                    "mpos_scoreB": self.mpos_scoreB,
                },
                os.path.join(self.opt.checkpoints_dir, self.opt.name, "mpos_score_B" + ".pth"),
            )