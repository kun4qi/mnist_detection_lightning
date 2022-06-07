#mnistの1を正常画像、0を異常画像
train.pyで学習させてresultフォルダにモデルを保存、
predict.pyで0と1の画像を含んだtestデータに対して、異常度を計算し、aucrocかaccuracyのスコアを表示]

!python3 train.py -c config.json
!python3 predict.py -c config.json
で実行できる

できなかったこと
Anoganで推論するとき、generatorとdiscriminatorの二つのmodelを読み込ませるのですが、1つのモデルを読み込ませる方法しか見つからず、train.pyとは分けて行い、
推論はlightningではなく、pytorchでやってしまいました。
https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

気になること
generatorとdiscriminatorのモデルを保存するときに
generator_checkpoint = callbacks.ModelCheckpoint(
    filename=config.save.generator_savename,
    monitor="valid_loss_g")
discriminator_checkpoint = callbacks.ModelCheckpoint(
    filename=config.save.discriminator_savename,
    monitor="valid_loss_d")
みたいな感じで二つchackpointを置いたが、1つのcheckpointで管理できないのか、わかりませんでした。


もとのcolabのコード
https://colab.research.google.com/drive/1K4DM2H8DyVKx9j05k52qLi_Ze2pyWD6y#scrollTo=-jMfIhJC_5Gc
lightning 変更後のcolabコード
https://colab.research.google.com/drive/1uqAHLJ7qA6R9NMkKVNkPcnEQHW54NT5e




自分用メモ
predict.pyでややこしいコードも理由は
https://qiita.com/sarashina/items/bcfd8134ae823a14009a