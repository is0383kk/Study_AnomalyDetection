import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def load_cifar10(batch=128):
    train_loader = DataLoader(
        datasets.CIFAR10('../../data',
                         train=True,
                         download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                [0.5, 0.5, 0.5],  # RGB 平均
                                [0.5, 0.5, 0.5]   # RGB 標準偏差
                                )
                         ])),
        batch_size=batch,
        shuffle=True
    )
 
    test_loader = DataLoader(
        datasets.CIFAR10('../../data',
                         train=False,
                         download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.5, 0.5, 0.5],  # RGB 平均
                                 [0.5, 0.5, 0.5]  # RGB 標準偏差
                             )
                         ])),
        batch_size=batch,
        shuffle=True
    )

    return {'train': train_loader, 'test': test_loader}

if __name__ == '__main__':
    loader = load_cifar10()
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR10のクラス

    for images, labels in loader['train']:
        print(images.shape)  # torch.Size([128, 3, 32, 32])

        # 試しに50枚を 5x10 で見てみる
        for i in range(5):
            for j in range(10):
                image = images[i*10+j] / 2 + 0.5
                image = image.numpy()
                plt.subplot(5, 10, i*10+j + 1)
                plt.imshow(np.transpose(image, (1, 2, 0)))  # matplotlibではチャネルは第3次元
                
                # 対応するラベル
                plt.title(classes[int(labels[i*10+j])])
                
                # 軸目盛や値はいらないので消す
                plt.tick_params(labelbottom=False,
                                labelleft=False,
                                labelright=False,
                                labeltop=False,
                                bottom=False,
                                left=False,
                                right=False,
                                top=False)

        plt.show()
        break