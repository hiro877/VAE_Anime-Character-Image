# パッケージのimport
import glob
import os.path as osp
import random
from scipy.stats import norm
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import torch.nn.functional as F

DEBUG=False

def make_datapath_list(phase="train"):
    """
    データのパスを格納したリストを作成する。

    Parameters
    ----------
    phase : 'train' or 'few'
        入力するデータのpathを選択する

    Returns
    -------
    path_list : list
        データへのパスを格納したリスト
    """

    if phase=="train":
        rootpath = "/work/data/image/pricone/chara"
    elif phase=="few":
        rootpath = "/work/data/image/pricone/chara/few"

    target_path = osp.join(rootpath+'/**/*.png')
    print(target_path)

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class ImageNormalize():
    """
    今回はリサイズしてテンソルに変換する

    ↓一応下記機能も残している
    画像の前処理クラス。訓練時、検証時で異なる動作をする。
    画像のサイズをリサイズし、色を標準化する。
    訓練時はRandomResizedCropとRandomHorizontalFlipでデータオーギュメンテーションする。


    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.ToTensor(),  # テンソルに変換
                # transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),  # テンソルに変換
                # transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        return self.data_transform[phase](img)

# Datasetを作成する
class PriconeCharaDataset(data.Dataset):
    """
    画像のDatasetクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    file_list : リスト
        画像のパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    """

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''

        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]
        img = img.convert("L")  # グレースケール変換


        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])


        return img_transformed

DEBUG_ENCODER=False
class Encoder(nn.Module):
    def __init__(self, latent_size, image_size):
        super(Encoder, self).__init__()
        # super().__init__()
        self.H = int(image_size/2)
        self.W = int(image_size/2)
        self.latent_size = latent_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(128*self.H*self.W, 64)
        self.linear2 = nn.Linear(64, latent_size)
        self.linear3 = nn.Linear(64, latent_size)
        self.dropout1 = torch.nn.Dropout2d(p=0.2) 

    def forward(self, x):
        if(DEBUG_ENCODER):print("first: ",x.shape)
        x = F.relu(self.conv1(x))
        if(DEBUG_ENCODER):print(x.shape)
        x = F.relu(self.conv2(x))
        if(DEBUG_ENCODER):print(x.shape)
        x = F.relu(self.conv3(x))
        if(DEBUG_ENCODER):print(x.shape)
        x = F.relu(self.conv4(x))
        if(DEBUG_ENCODER):print(x.shape)
        # x = self.dropout1(x)
        # x = torch.flatten(x)
        x = x.view(x.shape[0],-1)
        if(DEBUG_ENCODER):print(x.shape)
        x = F.relu(self.linear1(x))
        if(DEBUG_ENCODER):print(x.shape)
        z_mean = self.linear2(x)
        if(DEBUG_ENCODER):print(z_mean.shape)
        z_log_var = self.linear3(x)
        return z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
      epsilon = torch.randn(z_mean.shape, device="cuda")
      return z_mean + epsilon * torch.exp(0.5*z_log_var)

DEBUG_DECODER=False
class Decoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.H = int(image_size/2)
        self.W = int(image_size/2)
        self.to_shape = (128, self.H, self.W)  # (C, H, W)
        self.linear = nn.Linear(2, 2*np.prod(self.to_shape))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.dropout1 = torch.nn.Dropout2d(p=0.2) 
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if(DEBUG_DECODER):print("Decoder")
        if(DEBUG_DECODER):print(x.shape)
        # x = F.relu(self.linear(x))
        x = self.leakyrelu(self.linear(x))
        if(DEBUG_DECODER):print(x.shape)
        x = x.view(-1, 256, self.H, self.W)  # reshape to (-1, C, H, W)
        if(DEBUG_DECODER):print(x.shape)
        # x = F.relu(self.deconv1(x))
        x = self.leakyrelu(self.deconv1(x))
        if(DEBUG_DECODER):print(x.shape)
        # x = F.relu(self.conv2(x))
        x = self.leakyrelu(self.conv2(x))
        if(DEBUG_DECODER):print(x.shape)
        x = self.conv(x)
        # x = self.dropout1(x)
        if(DEBUG_DECODER):print(x.shape)
        if(DEBUG_DECODER):print("Decoder end")
        x = torch.sigmoid(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_size, image_size):
        super(VAE, self).__init__()
        # super().__init__()
        self.encoder = Encoder(latent_size, image_size)
        self.decoder = Decoder(image_size)

    def forward(self, x, C=1.0, k=1):
        """Call loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            x (Variable or ndarray): Input variable.
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        z_mean, z_log_var = self.encoder(x)

        rec_loss = 0
        for l in range(k):
            z = self.encoder.sampling(z_mean, z_log_var)
            if(DEBUG):print("latent: ", z.shape)
            if(DEBUG):print("latent data: ", z)
            y = self.decoder(z)
            if(DEBUG):print(torch.flatten(y, start_dim=1).shape, torch.flatten(x, start_dim=1).shape)
            # rec_loss += F.binary_cross_entropy(y.view(16, -1), x.view(16, -1)) / k
            rec_loss += F.binary_cross_entropy(torch.flatten(y, start_dim=1), torch.flatten(x, start_dim=1)) / k

        kl_loss = C * (z_mean ** 2 + torch.exp(z_log_var) - z_log_var - 1) * 0.5
        kl_loss = torch.sum(kl_loss) / len(x)
        return rec_loss + kl_loss


def show_images(model, device, image_size, epoch=0):
    """Display a 2D manifold of the digits"""
    n = 4  # 15x15 digits
    figure = np.zeros((image_size * n, image_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            if device:
                z_sample = torch.tensor(z_sample.astype(np.float32)).to(device)
            # print(z_sample.shape)
            # print(z_sample)
            x_decoded = model.decoder(z_sample)
            if device:
                x_decoded = x_decoded.to(device)
            digit = x_decoded.reshape(image_size, image_size)
            # print(digit.shape)
            digit = digit.to('cpu').detach().numpy().copy()
            ### 1 複数画像表示用
            # figure[i * image_size: (i + 1) * image_size, j * image_size: (j + 1) * image_size] = digit
            plt.imshow(digit, cmap='Greys_r')
            plt.show()

    ### 1 複数画像表示用
    # plt.figure(figsize=(10, 10))
    # plt.axis('off')
    # plt.imshow(figure, cmap='Greys_r')
    # plt.show()
    # plt.savefig('output/vae_fewImage_size50/vae_{}.png'.format(epoch))


if __name__ == '__main__':
    train_list = make_datapath_list(phase="few")

    # 前処理パラメータ
    size = 90
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    OUTPUT_DIR="output"
    if not os.path.exists(OUTPUT_DIR):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(OUTPUT_DIR)


    # Datasetを作成
    train_dataset = PriconeCharaDataset(file_list=train_list, 
        transform=ImageNormalize(size, mean, std), phase='train')

    """  Datasetの動作確認  start """
    # index = 0
    # print(train_dataset.__getitem__(index).size())
    # print(train_dataset.__getitem__(index)[1])
    # plt.imshow(train_dataset.__getitem__(0)[0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dataset.__getitem__(1)[0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dataset.__getitem__(2)[0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dataset.__getitem__(3)[0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dataset.__getitem__(4)[0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dataset.__getitem__(5)[0], cmap='gray')
    # plt.show()
    # """  Datasetの動作確認  end  """

    # ミニバッチのサイズを指定
    batch_size = 1

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader}

    """  Dataloaderの動作確認  start """
    # batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
    # inputs= next(
    #     batch_iterator)  # 1番目の要素を取り出す
    # print(inputs.size())
    # inputs= next(
    #     batch_iterator)  # 1番目の要素を取り出す
    # print(inputs.size())
    # inputs= next(
    #     batch_iterator)  # 1番目の要素を取り出す
    # print(inputs.size())
    # inputs= next(
    #     batch_iterator)  # 1番目の要素を取り出す
    # print(inputs.size())
    """  Dataloaderの動作確認  end  """

    # prepare model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_size=2   #潜在変数
    model = VAE(latent_size, size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    model.train()

    num_epochs=5000
    epoch_avg_loss=0
    # epochのループ
    for epoch in range(num_epochs):
        avg_loss = 0
        cnt = 0

        # データローダーからミニバッチを取り出すループ
        for x in dataloaders_dict["train"]:
            cnt += 1

            x = x.to(device)
            model.zero_grad()

            loss = model(x)
            loss.backward()
            optimizer.step()

            avg_loss += loss.data
            epoch_avg_loss+=avg_loss
            interval = 10 if device is 'cuda' else 10
            # if cnt % interval == 0:
            #     epoch_detail = epoch
            #     print('epoch: {:.2f}, loss: {:.4f}'.format(epoch_detail,
            #                                             float(avg_loss/cnt)))

        if(epoch % 100) == 0:
            print('epoch: {:.2f}, loss: {:.4f}'.format(epoch,
                                                        float(epoch_avg_loss/100)))
            epoch_avg_loss=0
            # show_images(model, device, size, epoch)

    show_images(model, device, size, epoch)
        

