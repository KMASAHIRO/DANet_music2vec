import random
import tensorflow as tf
import museval
import itertools
import soundfile as sf
import pandas
import numpy as np
import scipy
import librosa

# 環境音1つを音楽に混ぜた訓練データを生成するジェネレータ
def generator_train1(noise_Fourier_list, music_Fourier_list, batch_size, log_eps=0.0001):
    noise_index1 = 0
    noise_index2 = 0
    music_index1 = 0
    music_index2 = 0

    mixture = list()
    ideal_mask = list()
    correct = list()

    while True:
        if music_index1 == len(music_Fourier_list):
            music_index1 = 0
            music_index2 = 0
            noise_index1 = 0
            noise_index2 = 0
            mixture = list()
            ideal_mask = list()
            correct = list()
            random.shuffle(noise_Fourier_list)
            random.shuffle(music_Fourier_list)

        try:
            if noise_Fourier_list[noise_index1].shape[-1] < 100:
                raise TypeError("The noise data is too short.")
            if music_Fourier_list[music_index1].shape[-1] < 100:
                raise TypeError("The music data is too short.")
        except TypeError as message:
            print(message)

        noise_input = noise_Fourier_list[noise_index1][:, noise_index2 * 100:(noise_index2 + 1) * 100]
        music_input = music_Fourier_list[music_index1][:, music_index2 * 100:(music_index2 + 1) * 100]
        noise_index2 += 1
        music_index2 += 1

        if (noise_index2 + 1) * 100 > noise_Fourier_list[noise_index1].shape[-1]:
            noise_index1 += 1
            noise_index2 = 0
            if noise_index1 == len(noise_Fourier_list):
                noise_index1 = 0

        if (music_index2 + 1) * 100 > music_Fourier_list[music_index1].shape[-1]:
            music_index1 += 1
            music_index2 = 0

        with np.errstate(all="raise"):
            mix = noise_input + music_input
            mix = np.log(np.abs(mix) + log_eps)

            noise_input = np.log(np.abs(noise_input) + log_eps)
            music_input = np.log(np.abs(music_input) + log_eps)

            mixture.append(mix)
            correct.append(np.transpose(np.asarray([music_input, noise_input]), axes=[1, 2, 0]))
            max = np.max(np.asarray([music_input, noise_input]), axis=0)
            music_ideal = np.logical_not(music_input - max).astype(np.float32)
            noise_ideal = np.logical_not(noise_input - max).astype(np.float32)

            if not np.any(music_ideal):
                mixture.pop()
                correct.pop()
                continue
            if not np.any(noise_ideal):
                mixture.pop()
                correct.pop()
                if music_index2 == 0:
                    if music_index1 == 0:
                        music_index1 = len(music_Fourier_list) - 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                    else:
                        music_index1 -= 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                else:
                    music_index2 -= 1
                continue
            ideal_mask.append([music_ideal, noise_ideal])

            if len(mixture) == batch_size:
                x_train1 = np.asarray(mixture)
                x_train2 = np.asarray(ideal_mask)
                y_train = np.asarray(correct)

                mixture = list()
                ideal_mask = list()
                correct = list()

                yield [x_train1, x_train2], y_train

# 環境音2つを音楽に混ぜた訓練データを生成するジェネレータ
def generator_train2(noise_Fourier_list, music_Fourier_list, batch_size, log_eps=0.0001):
    noise_index1 = 0
    noise_index2 = 0
    music_index1 = 0
    music_index2 = 0

    mixture = list()
    ideal_mask = list()
    correct = list()

    while True:
        if music_index1 == len(music_Fourier_list):
            music_index1 = 0
            music_index2 = 0
            noise_index1 = 0
            noise_index2 = 0
            mixture = list()
            ideal_mask = list()
            correct = list()
            random.shuffle(noise_Fourier_list)
            random.shuffle(music_Fourier_list)

        try:
            if noise_Fourier_list[noise_index1].shape[-1] < 100:
                raise TypeError("The noise data is too short.")
            if music_Fourier_list[music_index1].shape[-1] < 100:
                raise TypeError("The music data is too short.")
        except TypeError as message:
            print(message)

        noise_input = noise_Fourier_list[noise_index1][:, noise_index2 * 100:(noise_index2 + 1) * 100] + \
                      noise_Fourier_list[(noise_index1 + 1) % len(noise_Fourier_list)][:,
                      noise_index2 * 100:(noise_index2 + 1) * 100]

        music_input = music_Fourier_list[music_index1][:, music_index2 * 100:(music_index2 + 1) * 100]
        noise_index2 += 1
        music_index2 += 1

        if (noise_index2 + 1) * 100 > noise_Fourier_list[noise_index1].shape[-1]:
            noise_index1 += 1
            noise_index2 = 0
            if noise_index1 == len(noise_Fourier_list):
                noise_index1 = 0

        if (music_index2 + 1) * 100 > music_Fourier_list[music_index1].shape[-1]:
            music_index1 += 1
            music_index2 = 0
            if music_index1 == len(music_Fourier_list):
                music_index1 = 0
                noise_index1 = 0
                noise_index2 = 0

        with np.errstate(all="raise"):
            mix = noise_input + music_input
            mix = np.log(np.abs(mix) + log_eps)

            noise_input = np.log(np.abs(noise_input) + log_eps)
            music_input = np.log(np.abs(music_input) + log_eps)

            mixture.append(mix)
            correct.append(np.transpose(np.asarray([music_input, noise_input]), axes=[1, 2, 0]))
            max = np.max(np.asarray([music_input, noise_input]), axis=0)
            music_ideal = np.logical_not(music_input - max).astype(np.float32)
            noise_ideal = np.logical_not(noise_input - max).astype(np.float32)

            if not np.any(music_ideal):
                mixture.pop()
                correct.pop()
                continue
            if not np.any(noise_ideal):
                mixture.pop()
                correct.pop()
                if music_index2 == 0:
                    if music_index1 == 0:
                        music_index1 = len(music_Fourier_list) - 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                    else:
                        music_index1 -= 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                else:
                    music_index2 -= 1
                continue
            ideal_mask.append([music_ideal, noise_ideal])

            if len(mixture) == batch_size:
                x_train1 = np.asarray(mixture)
                x_train2 = np.asarray(ideal_mask)
                y_train = np.asarray(correct)

                mixture = list()
                ideal_mask = list()
                correct = list()

                yield [x_train1, x_train2], y_train

# 短時間フーリエ変換で使用する窓関数
def square_root_of_hann(M, sym=False):
  w = scipy.signal.windows.hann(M, sym)
  w = np.sqrt(w)
  return w

# 推論時に位相を持ったスペクトログラムを入力されたとき、前処理するレイヤ
class Preparation(tf.keras.layers.Layer):
  def __init__(self, log_eps,  *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.log_eps = log_eps

  def call(self, input, training):
    if training:
      return input
    else:
      model_input = tf.math.log(tf.math.abs(input) + self.log_eps)
      return model_input

# アトラクターを生成するレイヤ(基本的には訓練時はideal mask、推論時はkmeansを使う)
class Attractor(tf.keras.layers.Layer):
    def __init__(self, kmeans_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kmeans_func = kmeans_func
        self.is_kmeans = False

    def call(self, input, training):
        if training:
            if self.is_kmeans:
                attractor = self.kmeans_func(input[0])
                attractor = tf.convert_to_tensor(attractor)
            else:
                att_num = tf.einsum('Ncft,Nftk->Nck', input[0], input[1])
                att_denom = tf.math.reduce_sum(input[0], axis=[2, 3])  # batch_size, c
                att_denom = tf.reshape(att_denom, [-1, 2, 1])
                attractor = att_num / att_denom
        else:
            attractor = self.kmeans_func(input[0])
            attractor = tf.convert_to_tensor(attractor)

        return attractor

# maskを混合音声に掛けて音声を分離するレイヤ(推論時には混合音声が位相のある複素数のデータになる)
class Make_clean_reference(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def call(self, input, training):
    if training:
      clean_reference = tf.einsum('Nft,Nftc->Nftc',input[0],input[1])
      return clean_reference
    else:
      clean_reference = tf.einsum('Nft,Nftc->Nftc',tf.cast(input[0], dtype=tf.complex64),tf.cast(input[1], dtype=tf.complex64))
      return clean_reference

# DANetのモデル
class DANet(tf.keras.Model):
    def __init__(self, source_num, embed_ndim, batch_size, log_eps=0.0001, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.source_num = source_num
        self.embed_ndim = embed_ndim
        self.log_eps = log_eps
        self.batch_size = batch_size
        self.cluster_centers_list = np.ones(shape=(self.batch_size, self.source_num, self.embed_ndim))

        self.preparation = Preparation(self.log_eps)
        self.reshape = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.lstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.lstm4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True),
                                                   merge_mode='concat')
        self.embedding1 = tf.keras.layers.Dense(129 * self.embed_ndim)
        self.embedding2 = tf.keras.layers.Reshape((100, 129, self.embed_ndim))
        self.embedding3 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))
        self.make_attractor = Attractor(self.kmeans_predict)
        self.make_mask = tf.keras.layers.Lambda(lambda x: tf.einsum('Nftk,Nck->Nftc', x[0], x[1]))
        self.make_clean_reference = Make_clean_reference()

    def call(self, inputs, training):
        x1 = self.preparation(inputs[0], training)
        x1 = self.reshape(x1)
        x1 = self.lstm1(x1)
        x1 = self.lstm2(x1)
        x1 = self.lstm3(x1)
        x1 = self.lstm4(x1)
        x1 = self.embedding1(x1)
        x1 = self.embedding2(x1)
        x1 = self.embedding3(x1)
        attractor = self.make_attractor([inputs[1], x1], training)
        mask = tf.keras.activations.softmax(self.make_mask([x1, attractor]))
        clean_reference = self.make_clean_reference([inputs[0], mask], training)

        return clean_reference

    # 訓練時にkmeansを用いて訓練する関数
    def train_with_kmeans(self, generator, steps, epochs, ideal_epochs):
        loss_result = list()
        print("start training", end="", flush=True)
        for epoch in range(epochs):
            loss_epoch = list()
            if epoch == ideal_epochs:
                self.to_kmeans_train()
            for step in range(steps):
                train_x, train_y = next(generator)
                if ideal_epochs <= epoch:
                    self.kmeans_fit(train_x[0])
                loss = self.train_on_batch(x=train_x, y=train_y)
                loss_epoch.append(loss)
                print("\033[2K\033[G", end="", flush=True)
                print("steps: {}/{}".format(step+1, steps), "{:.2f}".format(np.mean(loss_epoch)), end="", flush=True)
            print("\033[2K\033[G", end="", flush=True)
            print("Epoch {}/{}".format(epoch + 1, epochs), "loss: {:.2f}".format(np.mean(loss_epoch)), sep=" ",
                  flush=True)
            print("Epoch{} has ended.".format(epoch+1), end="", flush=True)
            loss_result.append(np.mean(loss_epoch))
        return loss_result

    # kmeansを用いた訓練に切り替える関数
    def to_kmeans_train(self):
        self.make_attractor.is_kmeans = True

    # ideal maskを用いた訓練に切り替える関数
    def to_idealmask_train(self):
        self.make_attractor.is_kmeans = False

    def get_embedded_data(self, inputs, training):
        x1 = self.preparation(inputs, training)
        x1 = self.reshape(x1)
        x1 = self.lstm1(x1)
        x1 = self.lstm2(x1)
        x1 = self.lstm3(x1)
        x1 = self.lstm4(x1)
        x1 = self.embedding1(x1)
        x1 = self.embedding2(x1)
        output = self.embedding3(x1)

        return output

    # kmeansの訓練をする関数
    def kmeans_fit(self, inputs, max_iter=1000, random_seed=0):
        embedded_data = self.get_embedded_data(inputs, training=False)

        shape = embedded_data.shape
        embedded_data = np.reshape(embedded_data, newshape=(shape[0], shape[1] * shape[2], shape[3]))

        cluster_centers_list = list()

        for n in range(len(inputs)):
            X = embedded_data[n]
            random_state = np.random.RandomState(random_seed)

            cycle = itertools.cycle(range(self.source_num))
            labels = np.fromiter(itertools.islice(cycle, X.shape[0]), dtype=np.int)
            random_state.shuffle(labels)
            labels_prev = np.zeros(X.shape[0])
            cluster_centers = np.zeros((self.source_num, X.shape[1]))

            for i in range(max_iter):
                for k in range(self.source_num):
                    XX = X[labels == k, :]
                    cluster_centers[k, :] = XX.mean(axis=0)

                dist = ((X[:, :, np.newaxis] - cluster_centers.T[np.newaxis, :, :]) ** 2).sum(axis=1)
                labels_prev = labels
                labels = dist.argmin(axis=1)

                for k in range(self.source_num):
                    if not np.any(labels == k):
                        labels[np.random.choice(len(labels), 1)] = k

                if (labels == labels_prev).all():
                    break

            for k in range(self.source_num):
                XX = X[labels == k, :]
                cluster_centers[k, :] = XX.mean(axis=0)

            cluster_centers_list.append(cluster_centers)

        self.cluster_centers_list = np.asarray(cluster_centers_list)

    # kmeansの推論結果を出力する関数
    def kmeans_predict(self, input):
        return self.cluster_centers_list

    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.cluster_centers_list = np.ones(shape=(self.batch_size, self.source_num, self.embed_ndim))

    # 推論する関数
    def prediction(self, input):
        self.kmeans_fit(input)
        fake_ideal_mask = np.zeros(shape=(input.shape[0], self.source_num, 129, 100))
        result = self.predict([input, fake_ideal_mask], batch_size=len(input))
        return result

    # モデルの重みをロードする関数
    def loading(self, path):
        input1 = np.zeros(shape=(self.batch_size, 129, 100))
        input2 = np.zeros(shape=(self.batch_size, self.source_num, 129, 100))
        temp = self.predict(x=[input1, input2], batch_size=self.batch_size)
        self.load_weights(path)

# 環境音1つを音楽に混ぜたテストデータを生成する関数
def create_testdata1(noise_Fourier_list, music_Fourier_list, log_eps=0.0001):
    noise_index1 = 0
    noise_index2 = 0
    music_index1 = 0
    music_index2 = 0

    mixture = list()
    ideal_mask = list()
    correct = list()

    while True:
        try:
            if noise_Fourier_list[noise_index1].shape[-1] < 100:
                raise TypeError("The noise data is too short.")
            if music_Fourier_list[music_index1].shape[-1] < 100:
                raise TypeError("The music data is too short.")
        except TypeError as message:
            print(message)

        noise_input = noise_Fourier_list[noise_index1][:, noise_index2 * 100:(noise_index2 + 1) * 100]
        music_input = music_Fourier_list[music_index1][:, music_index2 * 100:(music_index2 + 1) * 100]
        noise_index2 += 1
        music_index2 += 1

        if (noise_index2 + 1) * 100 > noise_Fourier_list[noise_index1].shape[-1]:
            noise_index1 += 1
            noise_index2 = 0
            if noise_index1 == len(noise_Fourier_list):
                noise_index1 = 0

        if (music_index2 + 1) * 100 > music_Fourier_list[music_index1].shape[-1]:
            music_index1 += 1
            music_index2 = 0
            if music_index1 == len(music_Fourier_list):
                music_index1 = 0
                noise_index1 = 0
                noise_index2 = 0

        with np.errstate(all="raise"):
            mix = noise_input + music_input

            mixture.append(mix)
            correct.append(np.transpose(np.asarray([music_input, noise_input]), axes=[1, 2, 0]))
            max = np.max(np.asarray([music_input, noise_input]), axis=0)
            music_ideal = np.logical_not(music_input - max).astype(np.float32)
            noise_ideal = np.logical_not(noise_input - max).astype(np.float32)

            if not np.any(music_ideal):
                mixture.pop()
                correct.pop()
                continue
            if not np.any(noise_ideal):
                mixture.pop()
                correct.pop()
                if music_index2 == 0:
                    if music_index1 == 0:
                        music_index1 = len(music_Fourier_list) - 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                    else:
                        music_index1 -= 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                else:
                    music_index2 -= 1
                continue
            ideal_mask.append([music_ideal, noise_ideal])

            if music_index1 == 0 and music_index2 == 0:
                x_train1 = np.asarray(mixture)
                x_train2 = np.asarray(ideal_mask)
                y_train = np.asarray(correct)

                mixture = list()
                ideal_mask = list()
                correct = list()

                return [x_train1, x_train2], y_train

# 環境音2つを音楽に混ぜたテストデータを生成する関数
def create_testdata2(noise_Fourier_list, music_Fourier_list, log_eps=0.0001):
    noise_index1 = 0
    noise_index2 = 0
    music_index1 = 0
    music_index2 = 0

    mixture = list()
    ideal_mask = list()
    correct = list()

    while True:
        try:
            if noise_Fourier_list[noise_index1].shape[-1] < 100:
                raise TypeError("The noise data is too short.")
            if music_Fourier_list[music_index1].shape[-1] < 100:
                raise TypeError("The music data is too short.")
        except TypeError as message:
            print(message)

        noise_input = noise_Fourier_list[noise_index1][:, noise_index2 * 100:(noise_index2 + 1) * 100] + \
                      noise_Fourier_list[(noise_index1 + 1) % len(noise_Fourier_list)][:,
                      noise_index2 * 100:(noise_index2 + 1) * 100]
        music_input = music_Fourier_list[music_index1][:, music_index2 * 100:(music_index2 + 1) * 100]
        noise_index2 += 1
        music_index2 += 1

        if (noise_index2 + 1) * 100 > noise_Fourier_list[noise_index1].shape[-1]:
            noise_index1 += 1
            noise_index2 = 0
            if noise_index1 == len(noise_Fourier_list):
                noise_index1 = 0

        if (music_index2 + 1) * 100 > music_Fourier_list[music_index1].shape[-1]:
            music_index1 += 1
            music_index2 = 0
            if music_index1 == len(music_Fourier_list):
                music_index1 = 0
                noise_index1 = 0
                noise_index2 = 0

        with np.errstate(all="raise"):
            mix = noise_input + music_input

            mixture.append(mix)
            correct.append(np.transpose(np.asarray([music_input, noise_input]), axes=[1, 2, 0]))
            max = np.max(np.asarray([music_input, noise_input]), axis=0)
            music_ideal = np.logical_not(music_input - max).astype(np.float32)
            noise_ideal = np.logical_not(noise_input - max).astype(np.float32)

            if not np.any(music_ideal):
                mixture.pop()
                correct.pop()
                continue
            if not np.any(noise_ideal):
                mixture.pop()
                correct.pop()
                if music_index2 == 0:
                    if music_index1 == 0:
                        music_index1 = len(music_Fourier_list) - 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                    else:
                        music_index1 -= 1
                        music_index2 = music_Fourier_list[music_index1].shape[-1] // 100 - 1
                else:
                    music_index2 -= 1
                continue
            ideal_mask.append([music_ideal, noise_ideal])

            if music_index1 == 0 and music_index2 == 0:
                x_train1 = np.asarray(mixture)
                x_train2 = np.asarray(ideal_mask)
                y_train = np.asarray(correct)

                mixture = list()
                ideal_mask = list()
                correct = list()

                return [x_train1, x_train2], y_train

# 損失関数
def loss_function(y_true, y_pred):
  frequency = tf.shape(y_true)[1]
  time = tf.shape(y_true)[2]
  frequency = tf.cast(frequency, tf.float32)
  time = tf.cast(time, tf.float32)
  return tf.reduce_sum((y_true - y_pred)**2) / (frequency*time)

# モデルを構築する関数
def create_model(source_num=2, embed_ndim=20, optimizer=None, loss=loss_function):
    batch_size = 30
    model = DANet(source_num=source_num, embed_ndim=embed_ndim, batch_size=batch_size, log_eps=0.0001)

    if optimizer is None:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=51450,
                                                                     decay_rate=0.03)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule), loss=loss)
    else:
        model.compile(optimizer=optimizer, loss=loss)

    input1 = np.zeros(shape=(batch_size, 129, 100))
    input2 = np.zeros(shape=(batch_size, source_num, 129, 100))
    temp = model.predict(x=[input1, input2], batch_size=batch_size)

    return model