import os
import torch
from functools import partial

import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss2, StylePatchNCELoss2
import util.util as util
from .RegGAN import Reg
from .networks import init_net, Upsample2


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
        parser.add_argument('--lambda_R', type=float, default=0.1, help='weight for frame consistency loss')
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
        self.loss_names = ['G_GAN', 'D', 'G', 'NCE', 'Reg']

        if self.isTrain:
            # self.visual_names = ['real_A1', 'fake_B1',
            #                      'real_A2', 'fake_B2',
            #                      'real_A3', 'fake_B3']
            self.visual_names = ['real_A1', 'fake_B1', 'reg_B1', 'real_R1',
                                 'real_A2', 'fake_B2', 'reg_B2', 'real_R2',
                                 'real_A3', 'fake_B3', 'reg_B3', 'real_R3']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B', ]
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            # self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'R']
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
        print(self.netG)

        if self.isTrain:
            opt.output_nc = 1
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netR = Reg(in_channel=2 * self.opt.num_K + 1).cuda()
            self.netR = init_net(self.netR, opt.init_type, opt.init_gain, opt.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionStyleNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss2(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=1e-4)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
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
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.optimizer_R.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        self.optimizer_R.step()

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

        if self.opt.isTrain:
            self.real_A = self.real_A.squeeze(1).permute(0, 3, 2, 1)
            self.real_B = self.real_B.squeeze(1).permute(0, 3, 2, 1)
            self.real_R = self.real_R.squeeze(1).permute(0, 3, 2, 1)

            # A
            self.real_A1 = self.real_A[:, [self.real_A.size(1) // 2 - 1]]
            self.real_A2 = self.real_A[:, [self.real_A.size(1) // 2]]
            self.real_A3 = self.real_A[:, [self.real_A.size(1) // 2 + 1]]
            # B
            self.real_B1 = self.real_B[:, [self.real_B.size(1) // 2 - 1]]
            self.real_B2 = self.real_B[:, [self.real_B.size(1) // 2]]
            self.real_B3 = self.real_B[:, [self.real_B.size(1) // 2 + 1]]

            # R
            self.real_R1 = self.real_R[:, [self.real_R.size(1) // 2 - 1]]
            self.real_R2 = self.real_R[:, [self.real_R.size(1) // 2]]
            self.real_R3 = self.real_R[:, [self.real_R.size(1) // 2 + 1]]
        else:
            self.image_meta = input['A_meta' if AtoB else 'B_meta']

        # self.real_A = torch.cat([self.real_A1, self.real_A2, self.real_A3], dim=1)
        # self.real_B = torch.cat([self.real_B1, self.real_B2, self.real_B3], dim=1)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        model = self.netG
        self.opt.patch_size = (256, 256, 3)
        model_inferer = partial(sliding_window_inference,
                                roi_size=(self.opt.patch_size[0], self.opt.patch_size[1], self.opt.patch_size[2]),
                                sw_batch_size=1, predictor=model, overlap=0.5)

        with torch.no_grad():
            print(self.real_A.shape)
            fake_A = model_inferer(self.real_A)
            print(fake_A.shape)
            # print(fake_A.min(), fake_A.max())
            # real_list = decollate_batch(self.real_A)
            # fake_list = decollate_batch(fake_A)
            #
            # real_list = [x.transpose(1, 3) for x in real_list]
            # fake_list = [x.transpose(1, 3) for x in fake_list]
            real_A_list = [torch.rot90(self.real_A.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]
            real_B_list = [torch.rot90(self.real_B.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]
            fake_A_list = [torch.rot90(fake_A.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]

        img_paths = self.get_image_paths()

        save_dir = os.path.join(self.opt.results_dir, self.opt.name, f'{self.opt.phase}_{self.opt.epoch}', 'images')

        os.makedirs(os.path.join(save_dir, 'real_A'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'fake_A'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'real_B'), exist_ok=True)

        for i in range(len(img_paths)):
            path = img_paths[i]
            name = os.path.basename(path)

            real = real_A_list[i][0]  # * 1000
            real_B = real_B_list[i][0]  # * 1000
            fake = fake_A_list[i][0]  # * 1000
            real = (real - real.min()) / (real.max() - real.min()) * 1000
            real_B = (real_B - real_B.min()) / (real_B.max() - real_B.min()) * 1000
            fake = (fake - fake.min()) / (fake.max() - fake.min()) * 1000

            # fake = (fake + 1) / 2 * 1000
            # real = (real + 1) / 2 * 1000

            nii = sitk.GetImageFromArray(fake.cpu().numpy().astype('int'))
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, 'fake_A', name))

            nii = sitk.GetImageFromArray(real.cpu().numpy().astype('int'))
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, 'real_A', name))

            nii = sitk.GetImageFromArray(real_B.cpu().numpy().astype('int'))
            nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            nii.SetDirection([x.item() for x in self.image_meta['direction']])
            sitk.WriteImage(nii, os.path.join(save_dir, 'real_B', name))

    def convert(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        model = self.netG
        self.opt.patch_size = (256, 256, 3)
        model_inferer = partial(sliding_window_inference,
                                roi_size=(self.opt.patch_size[0], self.opt.patch_size[1], self.opt.patch_size[2]),
                                sw_batch_size=1, predictor=model, overlap=0.5)

        # with torch.no_grad():
        #     print(self.real_A.shape)
        #     fake_A = model_inferer(self.real_A)
        #     print(fake_A.shape)
        #     # print(fake_A.min(), fake_A.max())
        #     # real_list = decollate_batch(self.real_A)
        #     # fake_list = decollate_batch(fake_A)
        #     #
        #     # real_list = [x.transpose(1, 3) for x in real_list]
        #     # fake_list = [x.transpose(1, 3) for x in fake_list]
        self.real_A = torch.rot90(self.real_A.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))
        self.real_B = torch.rot90(self.real_B.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))
        #     fake_A_list = [torch.rot90(fake_A.squeeze(1).transpose(1, 3), k=2, dims=(2, 3))]

        img_paths = self.get_image_paths()

        save_dir = os.path.join(self.opt.results_dir, self.opt.name, f'{self.opt.phase}_{self.opt.epoch}', 'images')

        os.makedirs(os.path.join(save_dir, 'real_A_png'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'fake_A'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'real_B_png'), exist_ok=True)

        for i in range(len(img_paths)):
            path = img_paths[i]
            name = os.path.basename(path)

            real_A = (self.real_A - self.real_A.min()) / (self.real_A.max() - self.real_A.min()) * 255
            real_B = (self.real_B - self.real_B.min()) / (self.real_B.max() - self.real_B.min()) * 255

            # fake = (fake + 1) / 2 * 1000
            # real = (real + 1) / 2 * 1000


            nii_A = real_A.cpu().numpy().astype('uint8')
            nii_B = real_B.cpu().numpy().astype('uint8')
            # print
            # nii_A = sitk.GetImageFromArray(real_A.cpu().numpy().astype('int'))
            # nii_B = sitk.GetImageFromArray(real_B.cpu().numpy().astype('int'))
            import cv2
            from PIL import Image
            for j in range(nii_A.shape[1]):
                img_A = Image.fromarray(nii_A[:, j].squeeze()).convert('L')
                img_A.save(os.path.join(save_dir, 'real_A_png', name.replace('.nii.gz', f'_{j}.png')))
                img_B = Image.fromarray(nii_B[:, j].squeeze()).convert('L')
                img_B.save(os.path.join(save_dir, 'real_B_png', name.replace('.nii.gz', f'_{j}.png')))
                # print(nii_A.shape)
                # print(nii_A[:, :, :, :, j].squeeze().shape)
                # print(nii_A[:, :, :, :, j].squeeze().min())
                # print(nii_A[:, :, :, :, j].squeeze().max())
                # print(name.replace('.nii.gz', f'_{j}.png'))
                # os.makedirs(os.path.join(save_dir, 'real_A_png'), exist_ok=True)
                # os.makedirs(os.path.join(save_dir, 'real_B_png'), exist_ok=True)
                # cv2.imwrite(os.path.join(save_dir, 'real_A_png', name.replace('.nii', f'_{j}.jpg')), nii_A[:, :, :, :, j].squeeze(),)
                # cv2.imwrite(os.path.join(save_dir, 'real_B_png', name.replace('.nii', f'_{j}.jpg')), nii_B[:, :, :, :, j].squeeze(),)
                # sitk.WriteImage(nii_A[:, :, :, :, j].squeeze(), os.path.join(save_dir, 'real_A_png', name.replace('.nii.gz', f'_{j}.png')))
                # sitk.WriteImage(nii_B[:, :, :, :, j].squeeze(), os.path.join(save_dir, 'real_B_png', name.replace('.nii.gz', f'_{j}.png')))

            # nii = sitk.GetImageFromArray(real_B.cpu().numpy().astype('int'))
            # nii.SetOrigin([x.item() for x in self.image_meta['origin']])
            # nii.SetSpacing([x.item() for x in self.image_meta['spacing']])
            # nii.SetDirection([x.item() for x in self.image_meta['direction']])
            # sitk.WriteImage(nii, os.path.join(save_dir, 'real_B', name))

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

        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size))
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B.reshape(-1, 1, self.opt.crop_size, self.opt.crop_size))
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

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
            self.loss_NCE_Y = self.criterionIdt(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # reg loss
        self.fake_reg, dvf = self.netR(self.fake_B, self.real_B)
        self.loss_Reg = self.opt.lambda_R * self.calculate_reg_loss((self.fake_reg, dvf), self.real_B)

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_Reg

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
