import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


#画像の前処理
data_transforms = {
    "train":transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]),
    "val":transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
#正規化をしない場合
# transforms.CenterCrop(28) # 画像サイズ
# transforms.RandomHorizontalFlip(), # ランダムに反転
# torchvision.transforms.ColorJitter(0,0,0,0),
# transforms.ToTensor()
to_tenser_transforms = transforms.Compose([
transforms.ToTensor() # Tensorに変換
])


root = "/home/is0383kk/workspace/study/datasets/MNIST"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,root,transform=None,train=True):
        train_classes = ["0","1","2","3","4","5","6","8","9"] #クラス名を定義
        test_classes = ["7"] #クラス名を定義
        """
        ただしディレクトリの仕様は以下の通り
        /root=上記/test/classes[i](0)/hoge.png
        /root=上記/train/classes[i](0など)/hogehoge.png
        """
        self.transform = transform
        #画像とラベルのリスト
        root_path = [] #教師データのディレクトリ名を格納するリスト
        images_path = [] #教師データの画像名を格納するリスト
        make_labels = [] #images_pathに対応するラベルを格納
        self.images = [] #画像データを全てリストに連結
        self.labels = [] #ラベルデータを全てリストに連結

        #訓練用とテスト用で分ける
        if train == True:
            for i in range(len(train_classes)):
                root_path.append(os.path.join(root,"train",train_classes[i]))
                images_path.append(os.listdir(root_path[i]))
                make_labels.append([i] * len(images_path[i]))
                #画像とラベルをそれぞれ１つのリストに格納する
                for image,label in zip(images_path[i],make_labels[i]):
                    self.images.append(os.path.join(root_path[i],image))
                    self.labels.append(label)
        else:
            for i in range(len(test_classes)):
                root_path.append(os.path.join(root,"test",test_classes[i]))
                images_path.append(os.listdir(root_path[i]))
                make_labels.append([i] * len(images_path[i]))
                for image,label in zip(images_path[i],make_labels[i]):
                    self.images.append(os.path.join(root_path[i],image))
                    self.labels.append(label)
        
    def __getitem__(self,index):
        #インデックスを元に画像のファイルパスとラベルを取得
        image = self.images[index]
        label = self.labels[index]
        #画像ファイルパスから画像を読み込む
        with open(image,"rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        #前処理がある場合
        if self.transform is not None:
            image = self.transform(image)
        #画像トラベルのペアを返す
        return image,label

    def __len__(self):
        #０からデータセットの合計値を返す
        return len(self.images)

#教師データの読み込み
print(root)
train_dataset = CustomDataset(root,to_tenser_transforms,train=True)
#print(custom_dataset)
"""引数の指定
root = データセットのパス
to_tenser_transforms = 画像の前処理(224 x 224)
"""
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=10,
                                            shuffle=True)

# テスト用
test_dataset = CustomDataset(root,to_tenser_transforms,train=False)
#print(custom_dataset)
"""引数の指定
root = データセットのパス
to_tenser_transforms = 画像の前処理(224 x 224)
"""
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=10,
                                            shuffle=True)
                                            
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)),interpolation="nearest")
    plt.show()


#for i in range(5):
#    print("TEST")
    #バッチサイズ分だけ画像とラベルの表示
for i,(images,labels) in enumerate(train_loader):
    print("i->",i)
    print("images->",images[i].size())
    print("labels->",labels[i])
    #print(labels.numpy())
    #print(type(images[0].numpy()))

    show(images[0])
    show(torchvision.utils.make_grid(images,padding=1))
    plt.axis("off")

    break
for i,(images,labels) in enumerate(test_loader):
    print("i->",i)
    print("images->",images[i].size())
    print("labels->",labels[i])
    #print(labels.numpy())
    #print(type(images[0].numpy()))

    show(images[0])
    show(torchvision.utils.make_grid(images,padding=1))
    plt.axis("off")

    break