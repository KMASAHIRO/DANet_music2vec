import random
import tensorflow as tf
import museval
import itertools
import soundfile as sf
import pandas
import numpy as np
import scipy
import librosa
import pickle
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')

from DANet_kmeans_module import generator_train1
from DANet_kmeans_module import square_root_of_hann
from DANet_kmeans_module import DANet
from DANet_kmeans_module import create_testdata1
from DANet_kmeans_module import loss_function
from DANet_kmeans_module import create_model

# セーブやロードに使うパス、訓練エポック数、ideal maskを使った訓練エポック数などの指定
esc_metafile = "ESC-50-master/meta/esc50.csv"
esc_audiofile = "ESC-50-master/audio/"
gtzan_path = "genres/"
test_data_path = "test_data.txt"
train_epochs = 20
train_ideal_epochs = 15
weight_save_path = "DANet_music_kmeans1_weights.h5"
SDR_path = "evaluation1.txt"
before1_file = "kmeans1_long_before1.wav"
before2_file = "kmeans1_long_before2.wav"
input_data_file = "kmeans1_long_input_data.wav"
after1_file = "kmeans1_long_after1.wav"
after2_file = "kmeans1_long_after2.wav"
mixed_png = "kmeans1_long_mixed.png"
before1_png = "kmeans1_long_before1.png"
before2_png = "kmeans1_long_before2.png"
after1_png = "kmeans1_long_after1.png"
after2_png = "kmeans1_long_after2.png"

# ESC-50の音声ファイルの準備
esc_meta = pandas.read_csv(esc_metafile)

esc_filenames = esc_meta.loc[:,"filename"].to_list()
for i in range(len(esc_filenames)):
  esc_filenames[i] = esc_audiofile + esc_filenames[i]
esc_filenames = np.asarray(esc_filenames)

# GTZANの音声ファイルの準備
genre_list = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
gtzan_filenames = list()
for genre in genre_list:
  for i in range(100):
    gtzan_filenames.append(gtzan_path + genre + "/" + genre + ".000{:02}.wav".format(i))
gtzan_filenames = np.asarray(gtzan_filenames)

# GTZANのラベルの準備
gtzan_labels = list()
for i in range(10):
  for k in range(100):
    gtzan_labels.append(tf.one_hot(i,10))
gtzan_labels = np.asarray(gtzan_labels)

# GTZANデータのシャッフル(最初はジャンルごとにソートされている)
np_random = np.random.RandomState(seed=1)
new_gtzan_filenames = np_random.permutation(gtzan_filenames)
np_random = np.random.RandomState(seed=1)
new_gtzan_labels = np_random.permutation(gtzan_labels)

# テストデータの記録(後で同じものをmusic2vecとつなげて使うため)
testdata_for_music2vec = {"esc_filenames":esc_filenames[1900:],"gtzan_filenames":new_gtzan_filenames[950:],\
                          "gtzan_labels":new_gtzan_labels[950:]}
with open(test_data_path, "wb") as f:
  pickle.dump(testdata_for_music2vec, f)

# モデル作成
model = create_model()

# 音楽データ(GTZAN)の短時間フーリエ変換
music_list = list()
count = 0
for name in new_gtzan_filenames:
  y,sr = librosa.load(name,sr=8000)
  D = librosa.stft(y, n_fft=256, hop_length=64, win_length=256, window=square_root_of_hann)
  music_list.append(D)
  count += D.shape[-1]//100

# 環境音データ(ESC-50)の短時間フーリエ変換
noise_list = list()
for name in esc_filenames:
  y,sr = librosa.load(name,sr=8000)
  D = librosa.stft(y, n_fft=256, hop_length=64, win_length=256, window=square_root_of_hann)
  noise_list.append(D)

# 訓練データ作成に使うジェネレータの作成
Generator = generator_train1(noise_Fourier_list=noise_list[:1900], music_Fourier_list=music_list[:950], batch_size=30)

# エポックごとのステップ数、エポック数、ideal maskを用いた訓練のエポック数の指定
steps = 1172
epochs = train_epochs
ideal_epochs = train_ideal_epochs

# 学習
model.train_with_kmeans(Generator, steps, epochs, ideal_epochs)

# 重みのセーブ
model.save_weights(weight_save_path, save_format='h5')

# テストデータでの推論
test_data, before_data = create_testdata1(noise_Fourier_list=noise_list[1900:],music_Fourier_list=music_list[950:])
result = model.prediction(test_data[0])

# 混合前の音声(教師データ)である音楽、環境音のスペクトログラムを音声波形に戻す
before = list()
for i in range(len(before_data)):
  before_i_1 = librosa.istft(before_data[i,:,:,0].reshape((129,100)), hop_length=64, win_length=256, window=square_root_of_hann)
  before_i_2 = librosa.istft(before_data[i,:,:,1].reshape((129,100)), hop_length=64, win_length=256, window=square_root_of_hann)
  before.append([before_i_1,before_i_2])
before = np.asarray(before)

# 入力データである混合音声のスペクトログラムを音声波形に戻す
mixed = list()
for i in range(len(before_data)):
  before_i_1 = before_data[i,:,:,0]
  before_i_2 = before_data[i,:,:,1]
  mixed_i = librosa.istft(before_i_1+before_i_2, hop_length=64, win_length=256, window=square_root_of_hann)
  mixed.append(mixed_i)
mixed = np.asarray(mixed)

# 推論結果のスペクトログラムを音声波形に戻す
after = list()
for i in range(len(result)):
  after_i_1 = librosa.istft(result[i,:,:,0].reshape((129,100)), hop_length=64, win_length=256, window=square_root_of_hann)
  after_i_2 = librosa.istft(result[i,:,:,1].reshape((129,100)), hop_length=64, win_length=256, window=square_root_of_hann)
  after.append([after_i_1,after_i_2])
after = np.asarray(after)

# GNSDR、GSIR、GSARを計算する
# 2つの出力の内、NSDR、SIR、SARの合計の高さによってどちらが音楽、環境音であるか決める
NSDR_list = list()
SIR_list = list()
SAR_list = list()
which = list()

for i in range(len(before)):
  reference = before[i]
  estimated = after[i]
  mix = np.asarray([mixed[i],mixed[i]])

  if not np.any(reference[0]):
    which.append(True)
    continue
  if not np.any(reference[1]):
    which.append(True)
    continue
  if not np.any(estimated[0]):
    which.append(False)
    continue
  if not np.any(estimated[1]):
    which.append(True)
    continue

  reference = reference.reshape(2,-1,1)
  estimated = estimated.reshape(2,-1,1)
  mix = mix.reshape(2,-1,1)

  SDR, ISR, SIR, SAR, perm = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=estimated)
  NSDR, _, _, _, _ = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=mix)

  NSDR = SDR - NSDR

  temp_NSDR = NSDR
  temp_SIR = SIR
  temp_SAR = SAR

  reference = before[i]
  estimated = after[i]
  mix = np.asarray([mixed[i],mixed[i]])

  one = estimated[0]
  two = estimated[1]
  estimated = np.asarray([two, one])

  reference = reference.reshape(2,-1,1)
  estimated = estimated.reshape(2,-1,1)
  mix = mix.reshape(2,-1,1)

  SDR, ISR, SIR, SAR, perm = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=estimated)
  NSDR, _, _, _, _ = museval.metrics.bss_eval(reference_sources=reference, estimated_sources=mix)

  NSDR = SDR - NSDR

  if np.sum(NSDR)+np.sum(SIR)+np.sum(SAR) > np.sum(temp_NSDR)+np.sum(temp_SIR)+np.sum(temp_SAR):
    which.append(False)
    NSDR_list.append(NSDR)
    SIR_list.append(SIR)
    SAR_list.append(SAR)
  else:
    which.append(True)
    NSDR_list.append(temp_NSDR)
    SIR_list.append(temp_SIR)
    SAR_list.append(temp_SAR)

NSDR_list = np.asarray(NSDR_list)
SIR_list = np.asarray(SIR_list)
SAR_list = np.asarray(SAR_list)
which = np.asarray(which)

GNSDR = np.mean(NSDR_list)
GSIR = np.mean(SIR_list)
GSAR = np.mean(SAR_list)


print("GNSDR:{:<4.2} GSAR:{:<4.2} GSIR:{:<4.2}".format(GNSDR, GSAR, GSIR))

# GNSDR、GSAR、GSIRに加え、教師データの順とテストデータでの出力順が同じ回数、逆の回数を記録
text = "GNSDR:{:<4.2} GSAR:{:<4.2} GSIR:{:<4.2}".format(GNSDR, GSAR, GSIR) + "\n" + \
  "same:{} reverse:{}".format(np.sum(which),np.sum(np.logical_not(which)))

with open(SDR_path, "w", encoding='UTF-8') as f:
  f.write(text)

# 混合前の音声(教師データ)、入力データの音声、分離後の音声を記録
long_before1 = np.concatenate(before[:37,0])
long_before2 = np.concatenate(before[:37,1])
long_mixed = np.concatenate(mixed[:37])
for i in range(37):
  if i:
    long_after1 = np.concatenate((long_after1,after[i,np.logical_not(which).astype(int)[i]]))
    long_after2 = np.concatenate((long_after2,after[i,which.astype(int)[i]]))
  else:
    long_after1 = after[i,np.logical_not(which).astype(int)[i]]
    long_after2 = after[i,np.logical_not(which).astype(int)[i]]

sf.write(before1_file, long_before1, 8000)
sf.write(before2_file, long_before2, 8000)
sf.write(input_data_file, long_mixed, 8000)
sf.write(after1_file, long_after1, 8000)
sf.write(after2_file, long_after2, 8000)

# 混合前の音声波形(教師データ)、入力データの音声波形、分離後の音声波形を記録
#mix
fig_mixed = plt.figure(figsize=(10,4))
plt.plot(np.arange(len(long_mixed))/8000, long_mixed)
fig_mixed.savefig(mixed_png)

#long_before1
fig_long_before1 = plt.figure(figsize=(10,4))
plt.plot(np.arange(len(long_before1))/8000, long_before1)
fig_long_before1.savefig(before1_png)

#long_before2
fig_long_before2 = plt.figure(figsize=(10,4))
plt.plot(np.arange(len(long_before2))/8000, long_before2)
fig_long_before2.savefig(before2_png)

#long_after1
fig_long_after1 = plt.figure(figsize=(10,4))
plt.plot(np.arange(len(long_after1))/8000, long_after1)
fig_long_after1.savefig(after1_png)

#long_after2
fig_long_after2 = plt.figure(figsize=(10,4))
plt.plot(np.arange(len(long_after2))/8000, long_after2)
fig_long_after2.savefig(after2_png)