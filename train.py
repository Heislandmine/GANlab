# todo:パラメータ探索を自動化できるよう改造
import tensorflow as tf
import models
import trainer
import os.path
import logger

# 設定
# todo:別モジュールとして切り出し
# todo:コンフィグファイルから読み込めるよう改造
g_w = "g_w.h5"
d_w = "d_w.h5"
image_h = 28
image_w = 28
image_c = None
# ハイパーパラメータ
# todo:別モジュールとして切り出し
# todo:コンフィグファイルから読み込めるよう改造
epochs = 100
batch_size = 32
noise_dim = 100
# モデルの生成
g = models.Generator()
d = models.Discriminator()
# ログ設定
logger = logger.Logger()
# 画像サイズの設定
if image_c is None:
    image_size = (None, image_h, image_w)
else:
    image_size = (None, image_h, image_w, image_c)
# 重みの復元
# todo:復元の要否をスイッチできるように改造
# todo:復元する重みファイルを指定できるようにする
if os.path.exists(g_w):
    g.build((None, noise_dim))
    g.load_weights(g_w)
if os.path.exists(d_w):
    d.build(image_size)
    d.load_weights(d_w)
# 学習の設定
t = trainer.Trainer(g, d, logger)
# データセットの構築
# todo:別ファイルに切り出し
# todo:コンフィグファイルからモデルを構築できるよう改造
(train, _), (_, _) = tf.keras.datasets.mnist.load_data()
train = train.reshape((train.shape[0], 28 * 28)).astype("float32")
train = (train - 127.5) / 127.5
dataset = tf.data.Dataset.from_tensor_slices(train)
dataset = dataset.batch(batch_size=batch_size)
dataset = dataset.shuffle(buffer_size=train.shape[0])
# 学習
t.train(dataset, epochs, batch_size)
# 学習結果の出力
# todo:別モジュールとして切り出し
