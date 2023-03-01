import os, shutil
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from pydub import AudioSegment
from pydub.utils import make_chunks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import layers
import pickle
from pydub.playback import play
import keras

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


POS = os.path.join("data","1")
NEG = os.path.join("data","0")

pos = tf.data.Dataset.list_files(POS+'\*.wav')
neg = tf.data.Dataset.list_files(NEG+'\*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

# lengths = []
# for file in os.listdir(os.path.join('data', '1')):
#     tensor_wave = load_wav_16k_mono(os.path.join('data', '1', file))
#     lengths.append(len(tensor_wave))

# print(tf.math.reduce_mean(lengths))
# print(tf.math.reduce_min(lengths))
# print(tf.math.reduce_max(lengths))

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:30000]
    zero_padding = tf.zeros([30000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

filepath, label = negatives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

train = data.take(240)
test = data.skip(240).take(len(data)-240)

samples, labels = train.as_numpy_iterator().next()

# (16, 1459, 257, 1)
# new_one = (16, 928, 257, 1)
# model = Sequential()
# model.add(Conv2D(16, (3,3), activation='relu', input_shape=(928, 257, 1)))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(Conv2D(16, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((3, 3)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
# print(model.summary())
# hist = model.fit(train, epochs=4, validation_data=test)
# model.save("wakeword.h5")

new_model = keras.models.load_model('wakeword.h5')

X_test, y_test = test.as_numpy_iterator().next()
yhat = new_model.predict(X_test)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
print(y_test.astype(int))
print(yhat)


