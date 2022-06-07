import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from pytorch_lightning import LightningDataModule

class image_data_set(Dataset):
    def __init__(self, data, image_size):
        self.images = data[:,:,:,None]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])

class MNISTDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_data_set = image_data_set

    def prepare_data(self):
        # download
        df = pd.read_csv(self.config.dataset.root_dir_path+"mnist_train_small.csv", dtype = np.float32)
        df.rename(columns={'6': 'label'}, inplace=True)
        # 学習データとして、1を2000枚使用する
        self.df = df.query("label in [1.0]").head(2000)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            df = self.df
            kf = KFold(n_splits=self.config.training.n_fold, shuffle=True, random_state=self.config.training.seed)
            df['fold']='nan'
            for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
              df.iloc[val_idx,-1] = fold
            df=df.reset_index(drop=True)
            df=df.reset_index(drop=True)
            df_train = df[df['fold']!=self.config.training.train_fold].reset_index(drop=True)
            df_valid = df[df['fold']==self.config.training.train_fold].reset_index(drop=True)
            # ラベル(1列目)を削除
            train = df_train.iloc[:,1:-1].values.astype('float32')
            valid = df_valid.iloc[:,1:-1].values.astype('float32')
            # 28×28 の行列に変換
            train = train.reshape(train.shape[0], 28, 28)
            valid = valid.reshape(valid.shape[0], 28, 28)

            self.train_dataset = self.image_data_set(train, image_size=self.config.dataset.img_size)
            self.valid_dataset = self.image_data_set(valid, image_size=self.config.dataset.img_size)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.dataset.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.dataset.valid_batch_size, shuffle=True)
