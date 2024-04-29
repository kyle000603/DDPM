import argparse
from Model.ddpm import DDPM
import run
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    device = torch.device("cuda")
    arg = argparse.ArgumentParser()
    arg.add_argument('--skip', action="store_true", default=False)
    arg.add_argument('--epoch', type=int, default=100)
    arg.add_argument('--batch', type=int, default=128)
    arg.add_argument('--lr', type=float, default=0.001)
    arg.add_argument('--tensorboard', type=str, default="DDPM_test")
    run.main(arg, device)