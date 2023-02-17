import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os
import torch.nn.functional as F
import pickle


class DINTSCycleGANModel(BaseModel):
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

        if not self.opt.DECODE:
            self.loss_names += ['dints_A', 'dints_B']

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
        from models.blocks.dints_monai import TopologySearch, TopologyInstance, DiNTS

        spatial_dims = 2
        channel_mul = 0.75  # 0.5
        num_depths = 3  # 4
        num_blocks = 6  # 12

        # train / test with full arch
        if not self.opt.DECODE:
            self.dints_space_A = TopologySearch(
                channel_mul=channel_mul,
                num_blocks=num_blocks,
                num_depths=num_depths,
                use_downsample=True,
                spatial_dims=spatial_dims,
                device='cuda:0',
            )

            self.dints_space_B = TopologySearch(
                channel_mul=channel_mul,
                num_blocks=num_blocks,
                num_depths=num_depths,
                use_downsample=True,
                spatial_dims=spatial_dims,
                device='cuda:0',
            )

            self.netG_A = DiNTS(
                dints_space=self.dints_space_A,
                spatial_dims=spatial_dims,
                in_channels=opt.input_nc,
                num_classes=1,
                use_downsample=True,
            ).cuda()

            self.netG_B = DiNTS(
                dints_space=self.dints_space_B,
                spatial_dims=spatial_dims,
                in_channels=opt.input_nc,
                num_classes=1,
                use_downsample=True,
            ).cuda()
        # train / test with part arch
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

            self.dints_space_A = TopologyInstance(
                channel_mul=channel_mul,
                num_blocks=num_blocks,
                num_depths=num_depths,
                use_downsample=True,
                spatial_dims=spatial_dims,
                arch_code=[arch_code_a, arch_code_c],
                device='cuda:0',
            )

            self.netG_A = DiNTS(
                dints_space=self.dints_space_A,
                spatial_dims=spatial_dims,
                in_channels=opt.input_nc,
                num_classes=1,
                node_a=node_a,
                use_downsample=True,
            ).cuda()

            ckptB = torch.load(os.path.join(self.opt.checkpoints_dir, self.opt.name, "search_code_B" + ".pth"))
            node_a = ckptB['node_a']
            arch_code_a = ckptB['arch_code_a']
            arch_code_c = ckptB['arch_code_c']

            print('Decoding arch of netG_B...')

            self.dints_space_B = TopologyInstance(
                channel_mul=channel_mul,
                num_blocks=num_blocks,
                num_depths=num_depths,
                use_downsample=True,
                spatial_dims=spatial_dims,
                arch_code=[arch_code_a, arch_code_c],
                device='cuda:0',
            )

            self.netG_B = DiNTS(
                dints_space=self.dints_space_B,
                spatial_dims=spatial_dims,
                in_channels=opt.input_nc,
                num_classes=1,
                node_a=node_a,
                use_downsample=True,
            ).cuda()

        # from .networks import init_net
        # init_net(self.netG_A, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids, initialize_weights=('stylegan2' not in self.netG_A))
        # init_net(self.netG_B, self.opt.init_type, self.opt.init_gain, self.opt.gpu_ids, initialize_weights=('stylegan2' not in self.netG_B))

        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)

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

            if not self.opt.DECODE:
                self.optimizer_GA_arch_a = torch.optim.Adam([self.dints_space_A.log_alpha_a], lr=opt.lr_arch,
                                                            betas=(opt.beta1, 0.999), weight_decay=0.0)
                self.optimizer_GA_arch_c = torch.optim.Adam([self.dints_space_A.log_alpha_c], lr=opt.lr_arch,
                                                            betas=(opt.beta1, 0.999), weight_decay=0.0)
                self.optimizer_GB_arch_a = torch.optim.Adam([self.dints_space_B.log_alpha_a], lr=opt.lr_arch,
                                                            betas=(opt.beta1, 0.999), weight_decay=0.0)
                self.optimizer_GB_arch_c = torch.optim.Adam([self.dints_space_B.log_alpha_c], lr=opt.lr_arch,
                                                            betas=(opt.beta1, 0.999), weight_decay=0.0)
                # self.optimizers.append(self.optimizer_G_arch_a)
                # self.optimizers.append(self.optimizer_G_arch_c)

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
            self.real_Aa = input['Aa' if AtoB else 'Ba'].to(self.device)
            self.real_Ba = input['Ba' if AtoB else 'Aa'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def forward_arch(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_Ba = self.netG_A(self.real_Aa)  # G_A(A)
        self.rec_Aa = self.netG_B(self.fake_Ba)  # G_B(G_A(A))
        self.fake_Aa = self.netG_B(self.real_Ba)  # G_B(B)
        self.rec_Ba = self.netG_A(self.fake_Aa)  # G_A(G_B(B))

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

    def backward_D_A_arch(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_Ba = self.fake_Ba_pool.query(self.fake_Ba)
        self.loss_D_Aa = self.backward_D_basic(self.netD_A, self.real_Ba, fake_Ba)

    def backward_D_B_arch(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_Aa = self.fake_A_pool.query(self.fake_Aa)
        self.loss_D_Ba = self.backward_D_basic(self.netD_B, self.real_Aa, fake_Aa)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
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

    def backward_G_arch(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_Aa = self.netG_A(self.real_Ba)
            self.loss_idt_Aa = self.criterionIdt(self.idt_Aa, self.real_Ba) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_Ba = self.netG_B(self.real_Aa)
            self.loss_idt_Ba = self.criterionIdt(self.idt_Ba, self.real_Aa) * lambda_A * lambda_idt
        else:
            self.loss_idt_Aa = 0
            self.loss_idt_Ba = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_Aa = self.criterionGAN_arch(self.netD_A(self.fake_Ba), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_Ba = self.criterionGAN_arch(self.netD_B(self.fake_Aa), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_Aa = self.criterionCycle(self.rec_Aa, self.real_Aa) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_Ba = self.criterionCycle(self.rec_Ba, self.real_Ba) * lambda_B
        # combined loss and calculate gradients
        self.loss_Ga = self.loss_G_Aa + self.loss_G_Ba + self.loss_cycle_Aa + self.loss_cycle_Ba + self.loss_idt_Aa + self.loss_idt_Ba

        if True:
            # A
            probs_a, arch_code_prob_a = self.dints_space_A.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()
            entropy_alpha_c = -(F.softmax(self.dints_space_A.log_alpha_c, dim=-1) *
                                F.log_softmax(self.dints_space_A.log_alpha_c, dim=-1)).mean()
            topology_loss = self.dints_space_A.get_topology_entropy(probs_a)

            ram_cost_full = self.dints_space_A.get_ram_cost_usage(self.real_A.shape, full=True)
            ram_cost_usage = self.dints_space_A.get_ram_cost_usage(self.real_A.shape)
            factor_ram_cost = 0.8
            ram_cost_loss = torch.abs(factor_ram_cost - ram_cost_usage / ram_cost_full)

            combination_weights = 1
            self.loss_dints_A = combination_weights * ((entropy_alpha_a + entropy_alpha_c) +
                                                       ram_cost_loss + 0.001 * topology_loss)
            # B
            probs_a, arch_code_prob_a = self.dints_space_B.get_prob_a(child=True)
            entropy_alpha_a = -((probs_a) * torch.log(probs_a + 1e-5)).mean()
            entropy_alpha_c = -(F.softmax(self.dints_space_B.log_alpha_c, dim=-1) *
                                F.log_softmax(self.dints_space_B.log_alpha_c, dim=-1)).mean()
            topology_loss = self.dints_space_B.get_topology_entropy(probs_a)

            ram_cost_full = self.dints_space_B.get_ram_cost_usage(self.real_A.shape, full=True)
            ram_cost_usage = self.dints_space_B.get_ram_cost_usage(self.real_A.shape)
            factor_ram_cost = 0.8
            ram_cost_loss = torch.abs(factor_ram_cost - ram_cost_usage / ram_cost_full)

            combination_weights = 1
            self.loss_dints_B = combination_weights * ((entropy_alpha_a + entropy_alpha_c) +
                                                       ram_cost_loss + 0.001 * topology_loss)

            self.loss_Ga += 0.5 * (self.loss_dints_A + self.loss_dints_B)
        else:
            self.loss_dints_A = 0
            self.loss_dints_B = 0

        self.loss_Ga.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        if not self.opt.DECODE:
            # # update weight of model parameters
            self.set_requires_grad([self.netG_A, self.netG_B], True)
            self.dints_space_A.log_alpha_a.requires_grad = False
            self.dints_space_A.log_alpha_c.requires_grad = False
            self.dints_space_B.log_alpha_a.requires_grad = False
            self.dints_space_B.log_alpha_c.requires_grad = False

        # forward
        self.forward()  # compute fake images and reconstruction images.

        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        if not self.opt.DECODE:
            # # update weight of model arch
            self.set_requires_grad([self.netG_A, self.netG_B], False)
            self.dints_space_A.log_alpha_a.requires_grad = True
            self.dints_space_A.log_alpha_c.requires_grad = True
            self.dints_space_B.log_alpha_a.requires_grad = True
            self.dints_space_B.log_alpha_c.requires_grad = True

            # forward_arch
            self.forward_arch()  # compute fake images and reconstruction images.

            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_GA_arch_a.zero_grad()
            self.optimizer_GA_arch_c.zero_grad()
            self.optimizer_GB_arch_a.zero_grad()
            self.optimizer_GB_arch_c.zero_grad()

            self.backward_G_arch()

            self.optimizer_GA_arch_a.step()
            self.optimizer_GA_arch_c.step()
            self.optimizer_GB_arch_a.step()
            self.optimizer_GB_arch_c.step()

            self.set_requires_grad([self.netG_A, self.netG_B], True)

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
            node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d = self.dints_space_A.decode()
            torch.save(
                {
                    "node_a": node_a_d,
                    "arch_code_a": arch_code_a_d,
                    "arch_code_a_max": arch_code_a_max_d,
                    "arch_code_c": arch_code_c_d,
                },
                os.path.join(self.opt.checkpoints_dir, self.opt.name, "search_code_A" + ".pth"),
            )

            node_a_d, arch_code_a_d, arch_code_c_d, arch_code_a_max_d = self.dints_space_B.decode()
            torch.save(
                {
                    "node_a": node_a_d,
                    "arch_code_a": arch_code_a_d,
                    "arch_code_a_max": arch_code_a_max_d,
                    "arch_code_c": arch_code_c_d,
                },
                os.path.join(self.opt.checkpoints_dir, self.opt.name, "search_code_B" + ".pth"),
            )
