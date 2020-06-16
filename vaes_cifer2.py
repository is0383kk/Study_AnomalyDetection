from __future__ import print_function
import argparse
import torch, torchvision
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from module import custom_dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
from module.data import MNIST
from module.data import CIFAR10

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--category', type=int, default=9, metavar='K',
                    help='how many category on datesets')
parser.add_argument('--anomaly', type=int, default=0, metavar='K',
                    help='Anomaly class')
parser.add_argument('--beta', type=bool, default=False, metavar='K',
                    help='set beta True or False')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)  

device = torch.device("cuda" if args.cuda else "cpu")




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

data_name = "cifar10"
if data_name == "cifar10":
    img_size=32
    nc=3
else:
    img_size=28
    nc=1
print(args.anomaly)
train_loader, anomaly_loader, img_size, nc = init_data_loader(
                                                    dataset=data_name, 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=args.batch_size, 
                                                    digits=args.anomaly
                                                    )
test_loader, _, img_size, nc = init_data_loader(
                                                    dataset=data_name, 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=args.batch_size,
                                                    train=False,
                                                    digits=args.anomaly
                                                    )
print("Train_loader", len(train_loader))

def prior(K, alpha):
    """
    Prior for the model.
    :param topics: number of topics
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    a = torch.Tensor(1, K).float().fill_(alpha) # 1 x 50 全て1.0
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # これを事前分布に定義

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)

class VAE_DIR(nn.Module):
    def __init__(self):
        super(VAE_DIR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 28, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(28, 56, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(56, 122, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(122, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 122, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(122, 56, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(56, 28, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(28, nc, kernel_size=5, stride=2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, args.category)
        self.fc22 = nn.Linear(128, args.category)

        self.fc3 = nn.Linear(args.category, 128)
        self.fc4 = nn.Linear(128, 256)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        # 事前分布のパラメータを定義
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(args.category, 0.3))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False


    def encode(self, x):
        conv = self.encoder(x);
        
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 256))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        z = F.softmax(z,dim=1)
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,256,1,1)
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

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_dir(self, recon_x, x, mu, logvar, K, beta):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # 事前分布の定義
        # 事前分布のパラメータを定義
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - K)
        #print(KLD)
        
        return BCE + (beta * KLD), -BCE

class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 28, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(28, 56, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(56, 122, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(122, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 122, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(122, 56, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(56, 28, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(28, nc, kernel_size=5, stride=2),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, 20)
        self.fc22 = nn.Linear(128, 20)

        self.fc3 = nn.Linear(20, 128)
        self.fc4 = nn.Linear(128, 256)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x);
        #print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 256))
        #print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        #print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1,256,1,1)
        #print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function_cnn(self, recon_x, x, mu, logvar, beta):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + (beta * KLD), -BCE


model_dir = VAE_DIR().to(device)
print(model_dir)
optimizer_dir = optim.Adam(model_dir.parameters(), lr=1e-3)

model_cnn = VAE_CNN().to(device)
optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=1e-3)


def train_dir(epoch, beta):
    model_dir.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer_dir.zero_grad()
        recon_batch, mu, logvar = model_dir(data)
        #print(f"recon_batch->{recon_batch}")
        loss, BCE = model_dir.loss_function_dir(recon_batch, data, mu, logvar, args.category, beta)
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        optimizer_dir.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss / len(train_loader.dataset), BCE

def test_dir(epoch, beta):
    model_dir.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model_dir(data)
            loss, BCE = model_dir.loss_function_dir(recon_batch, data, mu, logvar, args.category, beta)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                         'recon/dir/recon_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss.cpu().numpy(), BCE

def anomaly_dir(epoch, beta):
    model_dir.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(anomaly_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model_dir(data)
            loss, BCE = model_dir.loss_function_dir(recon_batch, data, mu, logvar, args.category, beta)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                         'recon/dir/anomaly_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Anomaly set loss: {:.4f}'.format(test_loss))
    return test_loss.cpu().numpy(), BCE

def train_cnn(epoch, beta):
    model_cnn.train()
    train_loss = 0
    loss = 0
    BCE = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        #print("label=>", _)
        optimizer_cnn.zero_grad()
        recon_batch, mu, logvar = model_cnn(data)
        print(f"recon_batch->{recon_batch.size()}")
        print(f"data->{data.size()}")
        #print(f"recon_batch->{recon_batch}")
        loss, BCE = model_cnn.loss_function_cnn(recon_batch, data, mu, logvar, beta)
        loss = loss.mean()
        loss.backward()
        #print(f"loss->{loss.item() / len(data)}")
        train_loss += loss.item()
        optimizer_cnn.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss / len(train_loader.dataset), BCE


def test_cnn(epoch, beta):
    model_cnn.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model_cnn(data)
            loss, BCE = model_cnn.loss_function_cnn(recon_batch, data, mu, logvar, beta)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                         'recon/cnn/recon_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss.cpu().numpy(), BCE

def anomaly_cnn(epoch, beta):
    model_cnn.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(anomaly_loader):
            print("Anomaly_label=>", _)
            data = data.to(device)
            recon_batch, mu, logvar = model_cnn(data)
            loss, BCE = model_cnn.loss_function_cnn(recon_batch, data, mu, logvar, beta)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                save_image(comparison.cpu(),
                         'recon/cnn/anomaly_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Anomaly set loss: {:.4f}'.format(test_loss))
    return test_loss.cpu().numpy(), BCE

if __name__ == "__main__":
    # CNN用
    tr_cnn_loss = []
    te_cnn_loss = []
    an_cnn_loss = []
    tr_cnn_bce = []
    te_cnn_bce = []
    an_cnn_bce = []
    # CNN_beta用
    tr_cnn_loss_b = []
    te_cnn_loss_b = []
    an_cnn_loss_b = []
    tr_cnn_bce_b = []
    te_cnn_bce_b = []
    an_cnn_bce_b = []
    # DIR用
    tr_dir_loss = []
    te_dir_loss = []
    an_dir_loss = []
    tr_dir_bce = []
    te_dir_bce = []
    an_dir_bce = []
    # DIR_beta用
    tr_dir_loss_b = []
    te_dir_loss_b = []
    an_dir_loss_b = []
    tr_dir_bce_b = []
    te_dir_bce_b = []
    an_dir_bce_b = []

    plt_epoch = np.arange(args.epochs)

    # グラフ用
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    
    
    ax1.set_xlabel('Epoch', fontsize=13)  
    ax1.set_ylabel('Loss', fontsize=13)  
    ax1.set_xticks(plt_epoch, minor=False)
    #ax1.set_ylim(80, 270)

    ax2.set_xlabel('Epoch', fontsize=13)  
    ax2.set_ylabel('ReconstructionError',fontsize=13)  
    ax2.set_xticks(plt_epoch, minor=False)
    
    ax3.set_xlabel('Epoch', fontsize=13)  
    ax3.set_ylabel('Loss',fontsize=13)  
    ax3.set_xticks(plt_epoch, minor=False)
    #ax3.set_ylim(80, 270)

    ax4.set_xlabel('Epoch', fontsize=13)  
    ax4.set_ylabel('ReconstructionError',fontsize=13) 
    ax4.set_xticks(plt_epoch, minor=False)
    
    ax5.set_xlabel('Epoch', fontsize=13)  
    ax5.set_ylabel('Loss',fontsize=13)  
    ax5.set_xticks(plt_epoch, minor=False)
    #ax5.set_ylim(80, 270)

    ax6.set_xlabel('Epoch', fontsize=13)  
    ax6.set_ylabel('ReconstructionError',fontsize=13)  
    ax6.set_xticks(plt_epoch, minor=False)

    c1, c2, c3 = "blue", "green", "red"
    l1b, l2b, l3b = "Train_baseline", "Test_baseline", "Anomaly_baseline"
    l1p, l2p, l3p = "Train_proposed", "Test_proposed", "Anomaly_proposed"
    l4b, l5b, l6b = "Train_baseline", "Test_baseline", "Anomaly_baseline"  
    l4p, l5p, l6p = "Train_proposed", "Test_proposed", "Anomaly_proposed"  
    
    
    # CNN学習
    for epoch in range(1, args.epochs + 1):
        if args.beta != True:
            print("no beta")
            trcl, trcbce = train_cnn(epoch, 1.0)
            tecl, tecbce = test_cnn(epoch, 1.0)
            ancl, ancbce = anomaly_cnn(epoch, 1.0)
        else:
            print("beta")
            trcl, trcbce = train_cnn(epoch, 10.0)
            tecl, tecbce = test_cnn(epoch, 10.0)
            ancl, ancbce = anomaly_cnn(epoch, 10.0)
        tr_cnn_loss.append(trcl)
        te_cnn_loss.append(tecl)
        tr_cnn_bce.append(trcbce)
        te_cnn_bce.append(tecbce)
        an_cnn_loss.append(ancl)
        an_cnn_bce.append(ancbce)
        if args.beta != True:
            torch.save(model_cnn.state_dict(), './pth/cnn_vae'+str(args.anomaly)+'.pth')
        else:
            torch.save(model_cnn.state_dict(), './pth/cnn_vae'+str(args.anomaly)+'_b.pth')

    # DIR学習
    for epoch in range(1, args.epochs + 1):
        if args.beta != True:
            trdl, trdbce = train_dir(epoch, 1.0)
            tedl, tedbce = test_dir(epoch, 1.0)
            andl, andbce = anomaly_dir(epoch, 1.0)
        else:
            trdl, trdbce = train_dir(epoch, 10.0)
            tedl, tedbce = test_dir(epoch, 10.0)
            andl, andbce = anomaly_dir(epoch, 10.0)
        tr_dir_loss.append(trdl)
        te_dir_loss.append(tedl)
        tr_dir_bce.append(trdbce)
        te_dir_bce.append(tedbce)
        an_dir_loss.append(andl)
        an_dir_bce.append(andbce)
        if args.beta != True:
            torch.save(model_dir.state_dict(), './pth/dir_vae'+str(args.anomaly)+'.pth')
        else:
            torch.save(model_dir.state_dict(), './pth/dir_vae'+str(args.anomaly)+'_b.pth')
        #print(f"{epoch} Epoch:Train Loss->{tr_loss}")
        #print(f"{epoch} Epoch:Test Loss->{te_loss}")
        #print(f"{epoch} Epoch:anomaly Loss->{an_loss}")
    np.save('./npy/dir_tr_loss.npy', np.array(tr_dir_loss))
    np.save('./npy/dir_te_loss.npy', np.array(te_dir_loss))
    np.save('./npy/dir_an_loss.npy', np.array(an_dir_loss))

    np.save('./npy/cnn_tr_loss.npy', np.array(tr_cnn_loss))
    np.save('./npy/cnn_te_loss.npy', np.array(te_cnn_loss))
    np.save('./npy/cnn_an_loss.npy', np.array(an_cnn_loss))
    # ロス関数プロット
    ax1.plot(plt_epoch, tr_cnn_loss, linestyle = "dashed", color=c1, label=l1b)
    ax1.plot(plt_epoch, te_cnn_loss, linestyle = "dashed", color=c2, label=l2b)
    #ax1.plot(plt_epoch, an_cnn_loss, linestyle = "dashed", color=c3, label=l3b)

    ax1.plot(plt_epoch, tr_dir_loss, color=c1, label=l1p)
    ax1.plot(plt_epoch, te_dir_loss, color=c2, label=l2p)
    #ax1.plot(plt_epoch, an_dir_loss, color=c3, label=l3p)
    
    ax1.legend(loc=1) 
    if args.beta != True:
        fig1.savefig('./result/'+str(args.anomaly)+'/loss.png')
    else:
        fig1.savefig('./result/'+str(args.anomaly)+'/loss_b.png')
    # 再構成項プロット
    ax2.plot(plt_epoch, tr_cnn_bce, linestyle = "dashed", color=c1, label=l4b)
    ax2.plot(plt_epoch, te_cnn_bce, linestyle = "dashed", color=c2, label=l5b)
    ax2.plot(plt_epoch, an_cnn_bce, linestyle = "dashed", color=c3, label=l6b)

    ax2.plot(plt_epoch, tr_dir_bce, color=c1, label=l4p)
    ax2.plot(plt_epoch, te_dir_bce, color=c2, label=l5p)
    ax2.plot(plt_epoch, an_dir_bce, color=c3, label=l6p)
    ax2.legend(loc='lower right') 
    if args.beta != True:
        fig2.savefig('./result/'+str(args.anomaly)+'/rec.png')
    else:
        fig2.savefig('./result/'+str(args.anomaly)+'/rec_b.png')
    # cnn単体ELBO
    ax3.plot(plt_epoch, tr_cnn_loss, linestyle = "dashed", color=c1, label=l1b)
    ax3.plot(plt_epoch, te_cnn_loss, linestyle = "dashed", color=c2, label=l2b)
    ax3.plot(plt_epoch, an_cnn_loss, linestyle = "dashed", color=c3, label=l3b)
    ax3.legend(loc=1) 
    if args.beta != True:
        fig3.savefig('./result/'+str(args.anomaly)+'/cnn_elbo.png')
    else:
        fig3.savefig('./result/'+str(args.anomaly)+'/cnn_elbo_b.png')

    # cnn単体recon
    ax4.plot(plt_epoch, tr_cnn_bce, linestyle = "dashed", color=c1, label=l4b)
    ax4.plot(plt_epoch, te_cnn_bce, linestyle = "dashed", color=c2, label=l5b)
    ax4.plot(plt_epoch, an_cnn_bce, linestyle = "dashed", color=c3, label=l6b)
    ax4.legend(loc='lower right') 
    if args.beta != True:
        fig4.savefig('./result/'+str(args.anomaly)+'/cnn_recon.png')
    else:
        fig4.savefig('./result/'+str(args.anomaly)+'/cnn_recon_b.png')

    # dir単体elbo
    ax5.plot(plt_epoch, tr_dir_loss, color=c1, label=l1p)
    ax5.plot(plt_epoch, te_dir_loss, color=c2, label=l2p)
    ax5.plot(plt_epoch, an_dir_loss, color=c3, label=l3p)
    ax5.legend(loc=1) 
    if args.beta != True:
        fig5.savefig('./result/'+str(args.anomaly)+'/dir_elbo.png')
    else:
        fig5.savefig('./result/'+str(args.anomaly)+'/dir_elbo_b.png')

    # dir単体recon
    ax6.plot(plt_epoch, tr_dir_bce, color=c1, label=l4p)
    ax6.plot(plt_epoch, te_dir_bce, color=c2, label=l5p)
    ax6.plot(plt_epoch, an_dir_bce, color=c3, label=l6p)
    ax6.legend(loc='lower right') 
    if args.beta != True:
        fig6.savefig('./result/'+str(args.anomaly)+'/dir_recon.png')
    else:
        fig6.savefig('./result/'+str(args.anomaly)+'/dir_recon_b.png')