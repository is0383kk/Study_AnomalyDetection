from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from module import custom_dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  

device = torch.device("cuda" if args.cuda else "cpu")

"""
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)
"""
# 独自に定義したデータローダの設定
to_tenser_transforms = transforms.Compose([
transforms.ToTensor() # Tensorに変換
])
train_test_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=True)

# shuffle せずに分割
"""
n_samples = len(train_test_dataset) # n_samples is 60000
train_size = int(n_samples * 0.8) # train_size is 48000
subset1_indices = list(range(0,train_size)) # [0,1,.....47999]
subset2_indices = list(range(train_size,n_samples)) # [48000,48001,.....59999]
train_dataset = Subset(train_test_dataset, subset1_indices)
test_dataset   = Subset(train_test_dataset, subset2_indices)
"""
# shuffleしてから分割してくれる.

n_samples = len(train_test_dataset) # n_samples is 60000
train_size = int(len(train_test_dataset) * 0.88) # train_size is 48000
test_size = n_samples - train_size # val_size is 48000
train_dataset, test_dataset = torch.utils.data.random_split(train_test_dataset, [train_size, test_size])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
anomaly_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=False)
anomaly_loader = torch.utils.data.DataLoader(dataset=anomaly_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False)
print(f"Train data->{len(train_dataset)}")
print(f"Test data->{len(test_dataset)}")
print(f"Anomaly data->{len(anomaly_dataset)}")

ngf = 64
ndf = 64
nc = 1 # 画像のチャンネル数
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
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
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)

        self.fc3 = nn.Linear(32, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x);
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 1024))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,1024,1,1)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    beta = 1.0
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (beta * BCE) + KLD, BCE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        #print(f"recon_batch->{recon_batch}")
        loss, BCE = loss_function(recon_batch, data, mu, logvar)
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss / len(train_loader.dataset), BCE


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, BCE = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 18)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'recon/cnn/recon_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss.cpu().numpy(), BCE

def anomaly(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(anomaly_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, BCE = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 18)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'recon/cnn/anomaly_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Anomaly set loss: {:.4f}'.format(test_loss))
    return test_loss.cpu().numpy(), BCE

if __name__ == "__main__":
    # グラフ用
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    ax1.set_xlabel('Epoch', fontsize=15)  
    ax1.set_ylabel('Loss', fontsize=15)  
    ax2.set_xlabel('Epoch', fontsize=15)  
    ax2.set_ylabel('ReconstructionError',fontsize=15)  
    ax3.set_xlabel('Epoch', fontsize=15)  
    ax3.set_ylabel('ReconstructionError',fontsize=15)  
    c1, c2, c3 = "blue", "green", "red"
    l1, l2, l3 = "Train_loss", "Test_loss", "Aomaly_loss"
    l4, l5, l6 = "Train_ReconErr", "Test_ReconErr", "Anomaly_ReconErr"  
    tr_loss = []
    te_loss = []
    an_loss = []
    tr_bce = []
    te_bce = []
    an_bce = []
    plt_epoch = np.arange(args.epochs)
    # 学習
    for epoch in range(1, args.epochs + 1):
        trl, trbce = train(epoch)
        tel, tebce = test(epoch)
        anl, anbce = anomaly(epoch)
        tr_loss.append(trl)
        te_loss.append(tel)
        tr_bce.append(trbce)
        te_bce.append(tebce)
        an_loss.append(anl)
        an_bce.append(anbce)
        
        #print(f"{epoch} Epoch:Train Loss->{tr_loss}")
        #print(f"{epoch} Epoch:Test Loss->{te_loss}")
        #print(f"{epoch} Epoch:anomaly Loss->{an_loss}")
        
        with torch.no_grad():
            sample = torch.randn(64, 32).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'recon/cnn/sample_' + str(epoch) + '.png')
    np.save('./npy/cnn_tr_loss.npy', np.array(tr_loss))
    np.save('./npy/cnn_te_loss.npy', np.array(te_loss))
    np.save('./npy/cnn_an_loss.npy', np.array(an_loss))
    # ロス関数プロット
    ax1.plot(plt_epoch, tr_loss, color=c1, label=l1)
    ax1.plot(plt_epoch, te_loss, color=c2, label=l2)
    ax1.plot(plt_epoch, an_loss, color=c3, label=l3)
    ax1.legend(loc=1) 
    fig1.savefig('cnn_loss.png')

    # 再構成項プロット
    ax2.plot(plt_epoch, tr_bce, color=c1, label=l4)
    ax2.plot(plt_epoch, te_bce, color=c2, label=l5)
    ax2.plot(plt_epoch, an_bce, color=c3, label=l6)
    ax2.legend(loc=1) 
    fig2.savefig('cnn_rec.png')