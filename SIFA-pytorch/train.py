from utils import UnpairedDataset, parse_config, get_config, set_random, UnpairedDataset2D
import yaml
import matplotlib.pyplot as plt
from model import SIFA
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib
import os
import configparser
from metrics import dice_eval


# matplotlib.use('TkAgg')

@torch.no_grad()
def val(net, validdata, device):
    net.eval()
    all_batch_dice = []
    # print('The number of val images = %d' % validdata_size)
    for i, (A, A_label, B, B_label) in enumerate(validdata):
        B = B.to(device)
        B_label = B_label.numpy().squeeze()
        output = net.test_seg(B).detach()
        output = output.squeeze()
        # B = B.detach().cpu().numpy()
        output = torch.argmax(output, dim=0)
        output = output.cpu().numpy()
        one_case_dice = dice_eval(output, B_label, 2)
        all_batch_dice += [one_case_dice]
    all_batch_dice = np.array(all_batch_dice)
    mean_dice = np.mean(all_batch_dice, axis=0)
    return mean_dice  # , dice1.value()[0], dice2.value()[0]


# train
def train():
    # load config
    config = "./config/train.cfg"
    config = parse_config(config)
    # load data
    print(config)

    A_path = config['train']['a_path']
    B_path = config['train']['b_path']
    batch_size = config['train']['batch_size']
    batch_size_val = config['test']['batch_size']

    trainset = UnpairedDataset(A_path, B_path, 'resize_and_crop', 'train')
    valset = UnpairedDataset(A_path, B_path, 'resize_and_crop', 'val')
    train_loader = DataLoader(trainset, batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size_val,
                            shuffle=True, drop_last=True)
    # load exp_name
    exp_name = config['train']['exp_name']

    loss_cycle = []
    loss_seg = []
    # load model

    device = torch.device('cuda:{}'.format(config['train']['gpu']))
    # device = torch.device('cpu')
    sifa_model = SIFA(config).to(device)
    # sifa_model = SIFA().to(device)
    # set train
    sifa_model.initialize()
    num_epochs = config['train']['num_epochs']
    save_epoch = num_epochs // 20
    bestDice = 0
    Dice = val(sifa_model, val_loader, device)
    print('Validation Dice Coeff: {}, bestDice: {}, bestepock: {}'.format(Dice, Dice, 0))
    for epoch in range(num_epochs):
        sifa_model.train()
        for i, (A, A_label, B, _) in enumerate(train_loader):
            A = A.to(device).detach()
            B = B.to(device).detach()
            A_label = A_label.to(device).detach().float()
            sifa_model.update_GAN(A, B)
            # print(A.shape)
            # print(B.shape)
            # print(A_label.shape, A_label.max(),A_label.min())
            sifa_model.update_seg(A, B, A_label)
            loss_cyclea, loss_cycleb, segloss = sifa_model.print_loss()
            loss_cycle.append(loss_cyclea + loss_cycleb)
            loss_seg.append(segloss)
            # Dice = val(sifa_model, val_loader,device)
            # print(Dice,"epoch-{}".format(epoch))
            # print(bestDice)
            print(epoch)
        # ddfseg_model.update_lr() #no need for changing lr
        Dice = val(sifa_model, val_loader, device)
        if bestDice < Dice:
            bestDice = Dice
            bestepoch = epoch
            print('saving the model at the end of epoch %d' % epoch)
            # if (epoch+1) % save_epoch == 0:
            model_dir = "save_model/" + str(exp_name)
            if (not os.path.exists(model_dir)):
                os.mkdir(model_dir)
            sifa_model.sample_image(epoch, exp_name)
            torch.save(sifa_model.state_dict(),
                       '{}/model-{}.pth'.format(model_dir, epoch + 1))
        sifa_model.update_lr()
        print('Validation Dice Coeff: {}, bestDice: {}, bestepock: {}'.format(Dice, bestDice, bestepoch))

    print('train finished')
    loss_cycle = np.array(loss_cycle)
    loss_seg = np.array(loss_seg)
    np.savez('trainingloss.npz', loss_cycle, loss_seg)
    x = np.arange(0, loss_cycle.shape[0])
    plt.figure(1)
    plt.plot(x, loss_cycle, label='cycle loss of training')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('cycle loss')
    plt.savefig('cycleloss.jpg')
    plt.close()
    plt.figure(2)
    plt.plot(x, loss_seg, label='seg loss of training')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('seg loss')
    plt.savefig('segloss.jpg')
    plt.close()
    print('loss saved')


if __name__ == '__main__':
    set_random()
    train()
