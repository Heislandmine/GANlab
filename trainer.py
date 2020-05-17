import tensorflow as tf
import time
import gen_image


class Trainer:
    def __init__(self, g, d):
        self.g = g
        self.d = d
        self.g_metrics = tf.keras.metrics.Mean()
        self.d_metrics = tf.keras.metrics.Mean()
        self.g_result = []
        self.d_result = []

    def train(self, dataset, epochs, batch_size):
        sample_noise = tf.random.normal([16, 100])  # todo:バッチサイズをパラメータとして設定できるようにする
        for epoch in range(epochs):
            start = time.time()
            self.g_metrics.reset_states()
            self.d_metrics.reset_states()
            for i, image_batch in enumerate(dataset):
                noise = tf.random.normal([batch_size, 100])
                self._train_step_g(noise)
                if i % 2 == 0:
                    self._train_step_d(image_batch, noise)
            print("epoch{}: {}".format(epoch + 1, time.time() - start))
            self.g_result.append(self.g_metrics.result())
            self.d_result.append(self.d_metrics.result())
            # 出力画像を保存
            gen_image.gen_image(self.g, sample_noise, str(epoch))
            if epoch % 100 == 0:
                self.g.save_weights("results\\weights//g_w_" + str(epoch) + ".h5")
                self.d.save_weights("results\\weights//d_w_" + str(epoch) + ".h5")

    @tf.function
    def _train_step_g(self, noise):
        with tf.GradientTape() as gen_tape:
            gen_image = self.g(noise)
            fake_output = self.d(gen_image)
            gen_loss = self.g.loss_function(tf.ones_like(fake_output), fake_output)

        g_gradient = gen_tape.gradient(gen_loss, self.g.trainable_variables)

        self.g.optimizer.apply_gradients(zip(g_gradient, self.g.trainable_variables))

        self.g_metrics(gen_loss)

    @tf.function
    def _train_step_d(self, image_batch, noise):
        with tf.GradientTape() as disc_tape:
            gen_image = self.g(noise)

            fake_output = self.d(gen_image)
            real_output = self.d(image_batch)

            fake_loss = self.d.loss_function(tf.zeros_like(fake_output), fake_output)
            real_loss = self.d.loss_function(tf.ones_like(real_output), real_output)
            total_loss = fake_loss + real_loss

        d_gradient = disc_tape.gradient(total_loss, self.d.trainable_variables)
        self.d.optimizer.apply_gradients(zip(d_gradient, self.d.trainable_variables))

        self.d_metrics(total_loss)