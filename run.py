import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
from Model.ddpm import DDPM
from Model.unet import UNet
from tqdm import tqdm
from Model import utils
from torch.utils.tensorboard import SummaryWriter


def prepare_data(batch_size, dataset="MNIST"):
    transform_c = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])
    transform_m = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[1.0])])
    if dataset == "MNIST":
        train_data = datasets.MNIST(root='./', train=True, download=True, transform=transform_m)
        test_data = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "CIFAR10":
        train_data = datasets.CIFAR10(root='./', train=True, download=True, transform=transform_c)
        test_data = datasets.CIFAR10(root='./', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=12, prefetch_factor=2**5)
    test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=12, prefetch_factor=2**5)

    return train_loader, test_loader


def train_epoch(train_loader:DataLoader, epoch:int, optimizer:torch.optim.Adam, model:DDPM, loss_function, device:torch.device, log:SummaryWriter):
    model.train()
    data_len = len(train_loader.dataset)
    tloader = tqdm(train_loader)
    tloader.set_description(f'Training... Epoch: {epoch:03d}')
    tloader.set_postfix_str(f'loss: {.0:0.4f}')
    loss_epoch = .0
    fin_idx = 0
    steps = model.steps
    for idx, [data, label] in enumerate(tloader):
        data = data.to(device)
        eta = torch.randn_like(data).to(device)
        n = len(data)
        t = torch.randint(0, steps, (n, )).to(device)
        optimizer.zero_grad()
    
        noisy_img = model(data, t, eta)
        eta_theta = model.backward(noisy_img, t.reshape(n, -1))
        loss = loss_function(eta_theta, eta)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item() * len(data)
        fin_idx = idx + 1
        tloader.set_postfix_str(f'loss: {loss_epoch/fin_idx:0.4f}')
    loss_epoch /= fin_idx
    log.add_scalar('loss/Train', loss_epoch, epoch)
    log.flush()

def test_epoch(test_loader:DataLoader, epoch:int, model:DDPM, loss_function, log:SummaryWriter, device:torch.device, STATUS="Testing"):
    model.eval()
    data_len = len(test_loader.dataset)
    vloader = tqdm(test_loader)
    vloader.set_description(f'{STATUS}... Epoch: {epoch:03d}')
    vloader.set_postfix_str(f'loss: {.0:0.4f}')
    loss_epoch = .0
    fin_idx = 0
    steps = model.steps
    for idx, [data, labels] in enumerate(vloader):
        data = data.to(device)
        n = len(data)
        eta = torch.randn_like(data).to(device)
        t = torch.randint(0, steps, (n, )).to(device)
        noisy_imgs = model(data, t, eta)

        eta_theta = model.backward(noisy_imgs, t.reshape(n, -1))
        loss = loss_function(eta_theta, eta)
        loss_epoch += loss.item() * len(data)
        fin_idx = idx + 1
        vloader.set_postfix_str(f'loss: {loss_epoch/fin_idx:0.4f}')
    loss_epoch /= fin_idx
    if STATUS == "Testing":
        log.add_scalar('loss/Test', loss_epoch, epoch)
    else:
        log.add_scalar('loss/Validation', loss_epoch, epoch)
    log.flush()
    if epoch % 2 == 0:
        torch.save(model.state_dict(), f'save/DDPM_{epoch}ep.pth')
    
    if STATUS == "Testing":
        generated = utils.generate_new_images(model, n_samples=100, device=device, gif_name='mnist_result.gif')
        utils.show_images(generated, "Final_result.jpg")


def get_tensorboard(log_name):
    logdir = os.path.join(os.getcwd(), "logs", log_name)
    return SummaryWriter(log_dir=logdir)

def train(train_loader:DataLoader, test_loader:DataLoader, model:DDPM, epochs:int, logdir:str, lr:float, device:torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()
    log = get_tensorboard(logdir)

    for epoch in range(1, epochs+1):
        train_epoch(train_loader=train_loader, epoch=epoch, optimizer=optimizer, model=model, loss_function=loss_function, log=log, device=device)

        with torch.no_grad():
            test_epoch(test_loader=test_loader, epoch=epoch, model=model, loss_function=loss_function, log=log, device=device, STATUS="Validating")
    
    with torch.no_grad():
        test_epoch(test_loader=test_loader, epoch=epoch, model=model, loss_function=loss_function, device=device, log=log)

def main(parser:argparse.ArgumentParser, device:torch.device):
    """
    arg.add_argument('--skip', action="store_true", default=False)
    arg.add_argument('--epoch', type=int, default=20)
    arg.add_argument('--batch', type=int, default=128)
    arg.add_argument('--lr', type=float, default=0.001)
    arg.add_argument('--tensorboard', type=str, default="DDPM_test")
    """
    args = parser.parse_args()
    model = DDPM(UNet(), device=device)
    train_loader, test_loader = prepare_data(batch_size=args.batch)
    utils.show_forward(model, test_loader, device)
    train(train_loader=train_loader, test_loader=test_loader, model=model, epochs=args.epoch, logdir=args.tensorboard, lr=args.lr, device=device)