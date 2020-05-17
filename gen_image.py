import models
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# todo:学習結果を出力できるよう改造
def gen_image(model, noise, filename="test"):
    image = model(noise).numpy()
    for i in range(image.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(image[i, :, :])
        plt.axis("off")
    plt.savefig(filename)


if __name__ == "__main__":
    g = models.Generator()
    g.build((None, 100))
    g.load_weights("g_w.h5")
    z = tf.random.normal((16, 100))
    y = g(z).numpy()
    fig = plt.figure(figsize=(4, 4))
    for i in range(y.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(y[i, :, :], cmap="gray")
        plt.axis("off")
    plt.savefig("results\\" + str(i))
    plt.show()
