import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import DataLoader

from models import Generator
from models import Discriminator
from utils import load_json
from utils import anomaly_score
from dataio import image_data_set
from train import ANOGAN

def main(config):
    print('read test data')
    df_test = pd.read_csv(config.dataset.root_dir_path+"mnist_test.csv",dtype = np.float32)
    df_test.rename(columns={'7': 'label'}, inplace=True)
    # テストデータとして、1、0の画像を合わせて500枚使用する
    df_test = df_test.query("label in [1.0, 0.0]").head(500)
    test = df_test.iloc[:,1:].values.astype('float32')
    test = test.reshape(test.shape[0], 28, 28)

    # GPU or CPU の指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_model = ANOGAN(config)
    g_model.load_state_dict(torch.load(config.save.load_model_dir + config.save.generator_savename + ".ckpt")["state_dict"])
    d_model = ANOGAN(config)
    d_model.load_state_dict(torch.load(config.save.load_model_dir + config.save.discriminator_savename+ ".ckpt")["state_dict"])
    G=g_model.generator.to(device).eval()
    D=d_model.discriminator.to(device).eval()

    test_set = image_data_set(test, image_size=config.dataset.img_size)
    test_loader = DataLoader(test_set, batch_size=config.dataset.test_batch_size, shuffle=False)
    bar = tqdm(enumerate(test_loader), total=len(test_loader))

    print('\n calculate anomality\n')
    for step, input_images in bar:
      input_images = input_images.to(device)
      # 潜在変数の初期化
      z = torch.randn(input_images.size(0), config.model.z_dim).to(device).view(input_images.size(0), config.model.z_dim, 1, 1).to(device)
      z.requires_grad = True
      # オプティマイザの定義
      z_optimizer = torch.optim.Adam([z], lr=1e-3)
      for epoch in range(5000):
        #z探し
        fake_images = G(z)
        loss, _, _ = anomaly_score(input_images, fake_images, D)
        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

      #異常度の計算
      fake_images = G(z)
      if step==0:
        _, anomality, _ = anomaly_score(input_images, fake_images, D)
        anomality = anomality.cpu().detach().numpy()
      else:
          _, anomality_temp, _ = anomaly_score(input_images, fake_images, D)
          anomality_temp = anomality_temp.cpu().detach().numpy()
          anomality = np.concatenate([anomality, anomality_temp])

    df_anomality=pd.DataFrame(anomality,columns = ['anomality'])
    df_test = pd.DataFrame(df_test['label'].reset_index(drop=True))
    df=pd.concat([df_test, df_anomality],axis=1)

    s=df['anomality'].min()
    e=df['anomality'].max()

    #accuracyが上がるしきい値を探索
    if config.test.accuracy:
        print('\nsearch best threshold on accuracy\n')
        list_acc = []
        for th in np.arange(s, e, 100.0):
          df=pd.concat([df_test, df_anomality],axis=1)
          df['judge'] = [1 if s > th else 0 for s in df['anomality']]
          df['label'] = [1 if s ==0 else 0 for s in df['label']]
          count=0
          for i,j in zip(df['label'],df['judge']):
            if i==j:
              count+=1
          accuracy=(count/len(df))*100
          list_acc.append((th, accuracy))

        ths,acc = sorted(list_acc, key=lambda x:x[1], reverse=True)[0]
        print(th,acc)

    #精度が上がるしきい値を探索
    if config.test.auc_roc:
        print('\nsearch best threshold on auc_roc\n')
        list_auc = []
        for th in np.arange(s, e, 100.0):
          df=pd.concat([df_test, df_anomality],axis=1)
          df['judge'] = [1 if s > th else 0 for s in df['anomality']]
          df['label'] = [1 if s ==0 else 0 for s in df['label']] 
          aucroc = roc_auc_score(df['label'].values, df['judge'].values)
          list_auc.append((th, aucroc))
        ths,auc = sorted(list_auc, key=lambda x:x[1], reverse=True)[0]
        print(ths,auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)

    main(config)