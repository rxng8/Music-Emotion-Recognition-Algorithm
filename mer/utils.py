
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

from .const import *

def get_spectrogram(waveform, input_len=44100):
  """ Check out https://www.tensorflow.org/io/tutorials/audio

  Args:
      waveform ([type]): Expect waveform array of shape (>44100,)
      input_len (int, optional): [description]. Defaults to 44100.

  Returns:
      Tensor: Spectrogram of the 1D waveform. Shape (freq, time, 1)
  """
  max_zero_padding = min(input_len, tf.shape(waveform))
  # Zero-padding for an audio waveform with less than 44,100 samples.
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      (input_len - max_zero_padding),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  """ Check out https://www.tensorflow.org/io/tutorials/audio

  Args:
      spectrogram ([type]): [description]
      ax (plt.axes[i]): [description]
  """
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def load_metadata(csv_folder):
  """ Pandas load multiple csv file and concat them into one df.

  Args:
      csv_folder (str): Path to the csv folder

  Returns:
      pd.DataFrame: The concatnated one!
  """
  global_df = pd.DataFrame()
  for i, fname in enumerate(os.listdir(csv_folder)):
    # headers: song_id, valence_mean, valence_std, arousal_mean, arousal_std
    df = pd.read_csv(os.path.join(csv_folder, fname), sep=r"\s*,\s*", engine="python")
    global_df = pd.concat([global_df, df], axis=0)
  
  # Reset the index
  global_df = global_df.reset_index(drop=True)

  return global_df

def split_train_test(df: pd.DataFrame, train_ratio: float):
  train_size = int(len(df) * train_ratio)
  train_df: pd.DataFrame = df[:train_size]
  train_df = train_df.reset_index(drop=True)
  test_df: pd.DataFrame = df[train_size:]
  test_df = test_df.reset_index(drop=True)
  return train_df, test_df

def plot_and_play(test_audio, second_id = 24.0, second_length = 1, channel = 0):
  """ Plot and play

  Args:
      test_audio ([type]): [description]
      second_id (float, optional): [description]. Defaults to 24.0.
      second_length (int, optional): [description]. Defaults to 1.
      channel (int, optional): [description]. Defaults to 0.
  """
  # Spectrogram of one second
  from_id = int(DEFAULT_FREQ * second_id)
  to_id = min(int(DEFAULT_FREQ * (second_id + second_length)), test_audio.shape[0])

  test_spectrogram = get_spectrogram(test_audio[from_id:, channel], input_len=int(DEFAULT_FREQ * second_length))
  print(test_spectrogram.shape)
  fig, axes = plt.subplots(2, figsize=(12, 8))
  timescale = np.arange(to_id - from_id)
  axes[0].plot(timescale, test_audio[from_id:to_id, channel].numpy())
  axes[0].set_title('Waveform')
  axes[0].set_xlim([0, int(DEFAULT_FREQ * second_length)])

  plot_spectrogram(test_spectrogram.numpy(), axes[1])
  axes[1].set_title('Spectrogram')
  plt.show()

  # Play sound
  sd.play(test_audio[from_id: to_id, channel], blocking=True)

def preprocess_waveforms(waveforms, input_len):
  """ Get the first input_len value of the waveforms, if not exist, pad it with 0.

  Args:
      waveforms ([type]): [description]
      input_len ([type]): [description]

  Returns:
      [type]: [description]
  """
  n_channel = waveforms.shape[-1]
  preprocessed = np.zeros((input_len, n_channel))
  if input_len <= waveforms.shape[0]:
    preprocessed = waveforms[:input_len, :]
  else:
    preprocessed[:waveforms.shape[0], :] = waveforms
  return tf.convert_to_tensor(preprocessed)