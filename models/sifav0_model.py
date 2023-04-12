import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch import nn
from tqdm import tqdm
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
from util.util import AvgrageMeter
from monai.losses import DiceLoss
from monai.transforms import Activations, AsDiscrete, Compose
import numpy as np


class SIFAV0Model(BaseModel):
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

            parser.add_argument('--lr_S', type=float, default=0.001,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--num_classes', type=int, default=2, help='')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
        self.loss_names += ['Dice_train', 'IoU_train', 'Dice_val', 'IoU_val']
        self.loss_Dice_train = torch.zeros(1)
        # self.loss_IoU_train = torch.zeros(1)
        self.loss_Dice_val = torch.zeros(1)
        # self.loss_IoU_val = torch.zeros(1)

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
        #     visual_names_A.append('idt_B')
        #     visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_Enc', 'G_Dec', 'S', 'D_A', 'D_B', 'D_S']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        opt.netG = 'resnet_sifa'
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        opt.netG = 'resnet_enc'
        self.netG_Enc = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        opt.netG = 'resnet_dec'
        self.netG_Dec = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.normG,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        self.netS = networks.define_G(opt.output_nc, opt.num_classes, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt=self.opt)
        # self.netS = SegNet(num_classes=opt.num_classes)

        if self.isTrain:  # define discriminators
            # opt.netD = 'basic_aux'
            opt.netD = 'basic'
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_S = networks.define_D(opt.num_classes, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            from util.loss_seg import DiceCeLoss
            self.criterionSeg = DiceCeLoss(self.opt.num_classes)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_Enc = torch.optim.Adam(self.netG_Enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_Dec = torch.optim.Adam(self.netG_Dec.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=opt.lr_S, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(),
                                                                self.netD_S.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_Enc)
            self.optimizers.append(self.optimizer_G_Dec)
            self.optimizers.append(self.optimizer_S)
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

        self.real_A_gt = input['A_gt'].to(self.device)
        self.real_B_gt = input['B_gt'].to(self.device)
        # print(torch.unique(self.real_A_gt), self.real_A_gt.shape)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)

        self.real_B_latent = self.netG_Enc(self.real_B)  # G_B(G_A(A))
        self.fake_B_latent = self.netG_Enc(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_Dec(self.real_B_latent, self.real_B)  # G_B(B)

        self.rec_A = self.netG_Dec(self.fake_B_latent, self.fake_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)

        self.real_B_SEG = self.netS(self.real_B_latent, tanh=False)
        self.fake_B_SEG = self.netS(self.fake_B_latent, tanh=False)

    def backward_D_basic(self, netD, real, fake, aux=False):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if aux:
            pred_real, pred_real_aux = netD(real)
            pred_fake, pred_fake_aux = netD(fake.detach())

            loss_D_real_aux = self.criterionGAN(pred_real_aux, True)
            loss_D_fake_aux = self.criterionGAN(pred_fake_aux, False)
            lambada = 0.25
        else:
            pred_real = netD(real)
            pred_fake = netD(fake.detach())
            loss_D_real_aux = 0
            loss_D_fake_aux = 0
            lambada = 0.5

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake + loss_D_real_aux + loss_D_fake_aux) * lambada
        loss_D.backward()

        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, torch.cat([fake_A, self.rec_A], dim=0))

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)

    def backward_D_S(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B_SEG = self.fake_B_SEG.detach()
        self.loss_D_B = self.backward_D_basic(self.netD_S, fake_B_SEG, self.real_B_SEG)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(torch.cat([self.fake_A, self.rec_A], dim=0)), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)
        #
        self.loss_G_S = self.criterionGAN(self.netD_S(self.real_B_SEG), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_S = self.criterionSeg(self.fake_B_SEG, self.real_A_gt)

        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_S + self.loss_cycle_A + self.loss_cycle_B + self.loss_S
        self.loss_G.backward()

    # def backward_S(self):
    #     """Calculate GAN loss for discriminator D_B"""
    #     loss_fake_B_seg = self.criterionSeg(self.fake_B_SEG, self.real_A_gt)
    #     loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
    #     loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
    #     loss_G_A = (self.criterionGAN(self.netD_A(self.fake_B), True) + self.criterionGAN(self.netD_A(self.fake_B), True)) * 0.5
    #     loss_G_S = self.criterionGAN(self.netD_S(self.fake_B_SEG), True)
    #     self.loss_D_B =

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_S], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G_A.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_G_Enc.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_G_Dec.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_S.zero_grad()  # set G_A and G_B's gradients to zero

        self.backward_G()  # calculate gradients for G_A and G_B

        self.optimizer_G_A.step()  # update G_A and G_B's weights
        self.optimizer_G_Enc.step()  # update G_A and G_B's weights
        self.optimizer_G_Dec.step()  # update G_A and G_B's weights
        self.optimizer_S.step()  # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_S], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.backward_D_S()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def val_metrics(self, epoch, train_dataloader, val_dataloader):
        self.netS.eval()

        dice_log = DiceMetric(include_background=False,
                              reduction=MetricReduction.MEAN_BATCH,
                              get_not_nans=True)
        # IoU_log = MeanIoU(include_background=False,
        #                   reduction=MetricReduction.MEAN_BATCH,
        #                   get_not_nans=True)
        # train
        dice_train = AvgrageMeter()
        dice_train.reset()
        # iou_train = AvgrageMeter()
        # iou_train.reset()

        # post_sigmoid = Activations(sigmoid=True)
        post_pred = AsDiscrete(argmax=True, to_onehot=self.opt.num_classes)
        post_label = AsDiscrete(to_onehot=self.opt.num_classes)

        for i, data in tqdm(enumerate(val_dataloader)):
            self.set_input(data)
            # reset
            dice_log.reset()
            # IoU_log.reset()

            fake_B_latent = self.netG_Enc(self.netG_A(self.real_A))
            real_A_SEG = self.netS(fake_B_latent, tanh=False)

            preds_list = [post_pred(x) for x in decollate_batch(real_A_SEG)]
            labels_list = [post_label(y) for y in decollate_batch(self.real_A_gt)]

            dice_log(y_pred=preds_list, y=labels_list)
            dice, not_nans = dice_log.aggregate()
            dice_train.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

            # iou
            # IoU_log(y_pred=preds_list, y=labels_list)
            # iou, not_nans = IoU_log.aggregate()
            # iou_train.update(iou.cpu().numpy(), n=not_nans.cpu().numpy())

        # val
        dice_val = AvgrageMeter()
        dice_val.reset()
        iou_val = AvgrageMeter()
        iou_val.reset()

        for i, data in tqdm(enumerate(val_dataloader)):
            self.set_input(data)
            # reset
            dice_log.reset()
            # IoU_log.reset()

            real_B_latent = self.netG_Enc(self.real_B)
            real_B_SEG = self.netS(real_B_latent, tanh=False)

            preds_list = [post_pred(x) for x in decollate_batch(real_B_SEG)]
            labels_list = [post_label(y) for y in decollate_batch(self.real_B_gt)]

            # dice
            dice_log(y_pred=preds_list, y=labels_list)
            dice, not_nans = dice_log.aggregate()
            dice_val.update(dice.cpu().numpy(), n=not_nans.cpu().numpy())

            # iou
            # IoU_log(y_pred=preds_list, y=labels_list)
            # iou, not_nans = IoU_log.aggregate()
            # iou_val.update(iou.cpu().numpy(), n=not_nans.cpu().numpy())

        self.loss_Dice_train = np.nanmean(dice_train.avg)
        # self.loss_IoU_train = np.nanmean(iou_train.avg)
        self.loss_Dice_val = np.nanmean(dice_val.avg)
        # self.loss_IoU_val = np.nanmean(iou_val.avg)

        print(f"EP: {epoch}, Train Dice: {self.loss_Dice_train:.4f} ({dice_train.avg}), "
              f"Val Dice: {self.loss_Dice_val:.4f} ({dice_val.avg})")
        self.netS.train()


class SegNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(num_classes),
            nn.Upsample(scale_factor=4, mode='bilinear')
        )

    def forward(self, x):
        return self.model(x)
