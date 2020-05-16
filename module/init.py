import numpy as np
import matplotlib.pyplot as plt
import os
import torch, torchvision

from data import MNIST
from data import CIFAR10

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


def make_dirs(*args):
	for dir in args:
		os.makedirs(dir, exist_ok=True)

def write_preprocessor(config):
	preprocessor = config
	print(vars(preprocessor))
	f = open(config.ckpt_path + "/preprocessor.dat", 'w')
	f.write(str(vars(preprocessor)))
	f.close()

img_size=32
num_channels=3
train_loader, anomaly_loader, img_size, nc = init_data_loader(
                                                    dataset="cifar10", 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=25, 
                                                    digits=6
                                                    )
test_loader, _, img_size, nc = init_data_loader(
                                                    dataset="cifar10", 
                                                    data_path="/home/is0383kk/workspace/study/data", 
                                                    batch_size=25,
                                                    train=False,
                                                    digits=6
                                                    )


#print(f"data_lodaer->{data_loader}")

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)),interpolation="nearest")
    plt.show()

for i,(images,labels) in enumerate(train_loader):
    print("i->",i)
    print("images->",images[i].size())
    print("labels->",labels)
    #print(labels.numpy())
    #print(type(images[0].numpy()))

    show(images[0])
    show(torchvision.utils.make_grid(images,padding=1))
    plt.axis("off")

    break
for i,(images,labels) in enumerate(anomaly_loader):
    print("i->",i)
    print("images->",images[i].size())
    print("labels->",labels)
    #print(labels.numpy())
    #print(type(images[0].numpy()))

    show(images[0])
    show(torchvision.utils.make_grid(images,padding=1))
    plt.axis("off")

    break

for i,(images,labels) in enumerate(test_loader):
    print("i->",i)
    print("images->",images[i].size())
    print("labels->",labels)
    #print(labels.numpy())
    #print(type(images[0].numpy()))

    show(images[0])
    show(torchvision.utils.make_grid(images,padding=1))
    plt.axis("off")

    break