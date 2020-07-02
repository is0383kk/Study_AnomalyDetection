# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset

from module.data import MNIST
from module.data import CIFAR10


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--temp', type=float, default=0.1, metavar='S',
                    help='tau(temperature) (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
                    help='hard Gumbel softmax')
parser.add_argument('--category', type=int, default=9, metavar='K',
                    help='how many category on datesets')
parser.add_argument('--anomaly', type=int, default=0, metavar='K',
                    help='Anomaly class')
parser.add_argument('--beta', type=bool, default=False, metavar='K',
                    help='set beta True or False')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
def init_data_loader(dataset, data_path, batch_size, train=True, digits=None):
    if dataset == "mnist":
        if digits is not None:
            return MNIST(data_path, batch_size, shuffle=False, train=train, condition_on=[digits])
        else:
            return MNIST(data_path, batch_size, shuffle=False, train=train)

    elif dataset == "cifar10":
        if digits is not None:
            return CIFAR10(data_path, batch_size, shuffle=False, train=train, condition_on=[digits])
        else:
            return CIFAR10(data_path, batch_size, shuffle=False, train=train)

train_loader, anomaly_loader, img_size, nc = init_data_loader(
                                                    dataset="mnist", 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=args.batch_size, 
                                                    digits=args.anomaly
                                                    )
test_loader, _, img_size, nc = init_data_loader(
                                                    dataset="mnist", 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=args.batch_size,
                                                    train=False,
                                                    digits=args.anomaly
                                                    )


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)

ngf = 64
ndf = 64
nc = 1 # 画像のチャンネル数
class VAE_gumbel(nn.Module):
    def __init__(self, temp):
        super(VAE_gumbel, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, latent_dim * categorical_dim)

        self.fc3 = nn.Linear(latent_dim * categorical_dim, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x);
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc2(h1)

    def decode(self, z):
        z = F.softmax(z,dim=1)
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def forward(self, x, temp, hard):
        #print(x.size())
        #print(x.view(-1,784).size())
        q = self.encode(x)
        q_y = q.view(q.size(0), latent_dim, categorical_dim)
        z = gumbel_softmax(q_y, temp, hard)
        #print("z=>",z)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())


latent_dim = 30
categorical_dim = args.category  # one-of-K vector

temp_min = 0.01
ANNEAL_RATE = 0.00003

model = VAE_gumbel(args.temp)
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy):
    BCE = F.binary_cross_entropy(recon_x, x,reduction='sum') / x.shape[0]
    #print(recon_x.size())
    #print(x.size())
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    temp = args.temp
    for batch_idx, (data, _) in enumerate(train_loader):
        #print("label",_)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp, args.hard)
        loss = loss_function(recon_batch, data, qy)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)
            print("temp",temp)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    temp = args.temp
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, args.hard)
        test_loss += loss_function(recon_batch, data, qy).item() * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'recon/cat/test_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def anomaly(epoch):
    model.eval()
    anomaly_loss = 0
    temp = args.temp
    for i, (data, _) in enumerate(anomaly_loader):
        #print("anomaly label=>",_)
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, args.hard)
        anomaly_loss += loss_function(recon_batch, data, qy).item() * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        if i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       'recon/cat/anomaly_' + str(epoch) + '.png', nrow=n)

    anomaly_loss /= len(anomaly_loader.dataset)
    print('====> Anomaly set loss: {:.4f}'.format(anomaly_loss))
    return anomaly_loss





if __name__ == '__main__':
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch', fontsize=13)  
    ax1.set_ylabel('Loss', fontsize=13)  
    plt_tr_loss = []
    plt_te_loss = []
    plt_an_loss = []
    plt_epoch = np.arange(args.epochs)
    for epoch in range(1, args.epochs + 1):
        tr_loss = train(epoch)
        te_loss = test(epoch)
        an_loss = anomaly(epoch)
        plt_tr_loss.append(tr_loss)
        plt_te_loss.append(te_loss)
        plt_an_loss.append(an_loss)
        if args.beta != True:
            torch.save(model.state_dict(), './pth/gs_vae'+str(args.anomaly)+'.pth')
        else:
            torch.save(model.state_dict(), './pth/gs_vae'+str(args.anomaly)+'_b.pth')
    np.save('./npy/gs_tr_loss.npy', np.array(tr_loss))
    np.save('./npy/gs_te_loss.npy', np.array(te_loss))
    np.save('./npy/gs_an_loss.npy', np.array(an_loss))
    #print(plt_epoch)
    #print(tr_loss)
    
    ax1.plot(plt_epoch, plt_tr_loss, linestyle = "dashed", color="blue", label="Train_loss")
    ax1.plot(plt_epoch, plt_te_loss, linestyle = "dashed", color="red", label="Test_loss")
    ax1.plot(plt_epoch, plt_an_loss, linestyle = "dashed", color="green", label="Anomaly_loss")
    ax1.legend(loc=1) 
    if args.beta != True:
        fig1.savefig('./result/'+str(args.anomaly)+'/loss.png')
    else:
        fig1.savefig('./result/'+str(args.anomaly)+'/loss_b.png')
