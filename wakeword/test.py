import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from pydub import AudioSegment
from pydub.utils import make_chunks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import layers
import keras
import numpy as np
from scripts.recording import record
import random
from pydub.playback import play

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

def preprocess(file_path): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:30000]
    zero_padding = tf.zeros([30000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


new_model = keras.models.load_model('models/wakeword.h5')
tony_files = os.listdir("Tony")


def real_time():
    audio = record()
    data = tf.data.Dataset.list_files("output.wav")
    data = data.map(preprocess)
    data = data.cache()
    data = data.shuffle(buffer_size=1000)
    data = data.batch(16)
    data = data.prefetch(8)
    yhat = new_model.predict(data)
    yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]
    print(yhat)
    file = random.choice(tony_files)
    song = AudioSegment.from_file(os.path.join("Tony", file))

    for x in yhat:
        if x == 1:
            play(song)
        else:
            pass

if __name__ == '__main__':
    while True:
        real_time()


