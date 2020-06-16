import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import argparse
from module.data import MNIST
from module.data import CIFAR10


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--anomaly', type=int, default=9, metavar='K',
                    help='Anomaly class')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x




learning_rate = 1e-3
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
                                                    dataset="cifar10", 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=128, 
                                                    digits=args.anomaly
                                                    )
test_loader, _, img_size, nc = init_data_loader(
                                                    dataset="cifar10", 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=args.batch_size,
                                                    train=False,
                                                    digits=args.anomaly
                                                    )
"""
#dataset = MNIST('../data', transform=img_transform)
#train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
to_tenser_transforms = transforms.Compose([
transforms.ToTensor() # Tensorに変換
])
train_test_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=True)

n_samples = len(train_test_dataset) # n_samples is 60000
train_size = int(len(train_test_dataset) * 0.88) # train_size is 48000
test_size = n_samples - train_size # val_size is 48000
train_dataset, test_dataset = torch.utils.data.random_split(train_test_dataset, [train_size, test_size])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
anomaly_dataset = custom_dataset.CustomDataset("/home/is0383kk/workspace/study/datasets/MNIST",to_tenser_transforms,train=False)
anomaly_loader = torch.utils.data.DataLoader(dataset=anomaly_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)

print(f"Train data->{len(train_dataset)}")
print(f"Test data->{len(test_dataset)}")
print(f"Anomaly data->{len(anomaly_dataset)}")
"""
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=1),  # b, 16, 10, 10
            #nn.Conv2d(3, 16, 4, stride=2, padding=1),
			nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 5, stride=2, padding=1),  # b, 8, 3, 3
            #nn.Conv2d(16, 8, 4, stride=2, padding=1)
			nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 5, stride=2),  # b, 16, 5, 5
            #nn.ConvTranspose2d(8, 16, 4, stride=2)
			nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1),  # b, 8, 15, 15
            #nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
			nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 6, stride=2, padding=1),  # b, 1, 28, 28
            #nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1)
			nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        #print(f"data->{data}")
        #print(f"_->{_}")
        data = data.to(device)
        optimizer.zero_grad()
        recon = model(data)
        print("recon",recon.size())
        print("data", data.size())
        loss = criterion(recon, data)
        #print(f"loss => {loss}")
        #loss = F.binary_cross_entropy(recon.view(-1, 784), data.view(-1, 784), reduction='sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    return loss.cpu()

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon = model(data)
            loss = criterion(recon, data)
            print(f"Test loss => {loss}")
            #loss = F.binary_cross_entropy(recon.view(-1, 784), data.view(-1, 784), reduction='sum')
            test_loss += loss.mean()
            test_loss.item()
            if i % args.log_interval == 0:
                n = min(data.size(0), 7)
                comparison = torch.cat([data[:n],
                                    recon[:n]])
                save_image(comparison.cpu(),
                        'recon/ae/recon_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {}'.format(test_loss))
    return test_loss.cpu()

def anomaly(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(anomaly_loader):
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            #print(f"Anomaly loss => {loss}")
            print("anomaly label",_)
            #loss = F.binary_cross_entropy(recon.view(-1, 784), data.view(-1, 784), reduction='sum')
            test_loss += loss.mean()
            test_loss.item()
            if i % args.log_interval == 0:
                n = min(data.size(0), 7)
                comparison = torch.cat([data[:n],
                                    recon[:n]])
                save_image(comparison.cpu(),
                        'recon/ae/anomaly_' + str(epoch) + '.png', nrow=n)
        

    test_loss /= len(test_loader.dataset)
    print('====> Anomaly set loss: {}'.format(test_loss))
    return test_loss.cpu().numpy()

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        tr_loss = train(epoch)
        te_loss = test(epoch)
        an_loss = anomaly(epoch)
        torch.save(model.state_dict(), './pth/cnn_ae'+str(args.anomaly)+'.pth')
        #test(epoch)
        
        

