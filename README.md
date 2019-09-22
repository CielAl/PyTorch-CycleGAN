# Pytorch-CycleGAN/StainGan [Forked]
Forked from aitorzip's CycleGan implementation: https://github.com/aitorzip/PyTorch-CycleGAN
* A clean and readable Pytorch implementation of CycleGAN (https://arxiv.org/abs/1703.10593)
* Simply add the PatchGan into the Discriminator models (from the stain-gan implementation https://github.com/xtarx/StainGAN). 
    Use --stain_gan flag to apply patch_gan discrminator.
* use --gpuid to specify the device in case there are multiple gpus.
