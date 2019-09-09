#!/usr/bin/python3

import argparse
import os
import sys

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import ImageDataset
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--gpuid', type=int, default=0, help='GPU ID')
opt, remaining = parser.parse_known_args()
print(opt)
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.cuda:
    device = torch.device(f"cuda:{opt.gpuid}")
else:
    device = torch.device("cpu")

# ##### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
# noinspection PyArgumentList
input_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size).to(device)
# noinspection PyArgumentList
input_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size).to(device)

# Dataset loader
transforms_ = [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
data_set = ImageDataset(opt.dataroot, transforms_=transforms_, mode='test')
data_loader = DataLoader(data_set,
                         batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
# ##################################

# ##### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')

for i, batch in enumerate(data_loader):
    # Set model input
    real_A = input_A.copy_(batch['A']).requires_grad_(True)
    real_B = input_B.copy_(batch['B']).requires_grad(True)

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'output/A/%04d.png' % (i + 1))
    save_image(fake_B, 'output/B/%04d.png' % (i + 1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(data_loader)))

sys.stdout.write('\n')
###################################
