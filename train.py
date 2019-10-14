#!/usr/bin/python3

import argparse
import itertools
from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import DataLoader

from datasets import ImageDataset
from models import Generator
from models import PatchGanDiscriminator
from utils import LambdaLR
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from utils import ReplayBuffer
from utils import weights_init_normal
from torchnet.meter import AverageValueMeter
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/tcga/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=25,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=6, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpuid', type=int, default=0, help='GPU ID')
parser.add_argument('--stain_gan', default=True, action='store_true', help='use Stain Gan Discriminator')
opt, remaining = parser.parse_known_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.cuda:
    device = torch.device(f"cuda:{opt.gpuid}")
else:
    device = torch.device("cpu")
print(device)
# ##### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
netD_A = PatchGanDiscriminator(opt.input_nc, norm_layer=nn.InstanceNorm2d).to(device)
netD_B = PatchGanDiscriminator(opt.output_nc, norm_layer=nn.InstanceNorm2d).to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# noinspection PyTypeChecker
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# noinspection PyTypeChecker
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# noinspection PyTypeChecker
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
# noinspection PyArgumentList
input_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)
# noinspection PyArgumentList
input_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)
# noinspection PyArgumentList
target_real = torch.Tensor(opt.batchSize).fill_(1.0).to(device)
# noinspection PyArgumentList
target_fake = torch.Tensor(opt.batchSize).fill_(0.0).to(device)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
data_set = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)
data_loader = DataLoader(data_set,
                         batch_size=opt.batchSize,
                         shuffle=True,
                         num_workers=opt.n_cpu)

loss_meters: Dict[str, AverageValueMeter] = {
    'loss_G_meter': AverageValueMeter(),
    'loss_G_identity_meter': AverageValueMeter(),
    'loss_G_GAN_meter': AverageValueMeter(),
    'loss_G_cycle_meter': AverageValueMeter(),
    'loss_D_meter': AverageValueMeter()
}
# Loss plot
# logger = Logger(opt.n_epochs, len(data_loader))
loss_logger = VisdomPlotLogger('line', opts={'title': 'Loss'})

real_A_im_logger = VisdomLogger('image', opts={'title': 'Real A'})
real_B_im_logger = VisdomLogger('image', opts={'title': 'Real B'})
fake_A_im_logger = VisdomLogger('image', opts={'title': 'Fake A'})
fake_B_im_logger = VisdomLogger('image', opts={'title': 'Fake B'})
###################################

# ##### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    print(f"Epoch:{epoch}")
    for i, batch in enumerate(tqdm(data_loader)):
        # Set model input

        real_A = batch['A'].to(device)  # input_A.copy_(batch['A']).clone().detach().requires_grad_(True)
        real_B = batch['B'].to(device)  # input_B.copy_(batch['B']).clone().detach().requires_grad_(True)
        # ##### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)

        loss_GAN_A2B = criterion_GAN(pred_fake.squeeze(), target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake.squeeze(), target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        # ##################################

        # ##### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        # ##### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake.squeeze(), target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        loss_meters['loss_G_meter'].add(loss_G.item())
        loss_meters['loss_G_identity_meter'].add((loss_identity_A + loss_identity_B).item())
        loss_meters['loss_G_GAN_meter'].add((loss_GAN_A2B + loss_GAN_B2A).item())
        loss_meters['loss_G_cycle_meter'].add((loss_cycle_ABA + loss_cycle_BAB).item())
        loss_meters['loss_D_meter'].add((loss_D_A + loss_D_B).item())
        real_A_im_logger.log(
            make_grid(real_A.detach().cpu(), nrow=int(opt.batchSize ** 0.5), normalize=True, range=(0, 1)).numpy())
        real_B_im_logger.log(
            make_grid(real_B.detach().cpu(), nrow=int(opt.batchSize ** 0.5), normalize=True, range=(0, 1)).numpy())
        fake_A_im_logger.log(
            make_grid(fake_A.detach().cpu(), nrow=int(opt.batchSize ** 0.5), normalize=True, range=(0, 1)).numpy())
        fake_B_im_logger.log(
            make_grid(fake_B.detach().cpu(), nrow=int(opt.batchSize ** 0.5), normalize=True, range=(0, 1)).numpy())
        ###################################

        # Progress report (http://localhost:8097)
        """
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                   images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        """
    # Update learning rates
    for loss_name, meter in loss_meters.items():
        loss_logger.log(meter.value(), name=loss_name)

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')
# ##################################
