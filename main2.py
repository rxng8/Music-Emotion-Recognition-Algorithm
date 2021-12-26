# %%

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import sounddevice as sd
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from mer.utils import get_spectrogram, \
  plot_spectrogram, \
  load_metadata, \
  plot_and_play, \
  preprocess_waveforms, \
  split_train_test

from mer.const import *
from mer.loss import simple_mse_loss
from mer.model import SimpleDenseModel

# Set the seed value for experiment reproducibility.
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

sd.default.samplerate = DEFAULT_FREQ

ANNOTATION_SONG_LEVEL = "./dataset/DEAM/annotations/annotations averaged per song/song_level/"
AUDIO_FOLDER = "./dataset/DEAM/wav"
filenames = tf.io.gfile.glob(str(AUDIO_FOLDER) + '/*')

weights_path = "./weights/simple_dense/checkpoint"
history_path = "./history/simple_dense.npy"

# Process with average annotation per song. 
df = load_metadata(ANNOTATION_SONG_LEVEL)

train_df, test_df = split_train_test(df, TRAIN_RATIO)

# test_file = tf.io.read_file(os.path.join(AUDIO_FOLDER, "2011.wav"))
# test_audio, _ = tf.audio.decode_wav(contents=test_file)
# test_audio.shape
# test_audio = preprocess_waveforms(test_audio, WAVE_ARRAY_LENGTH)
# test_audio.shape

# plot_and_play(test_audio, 24, 5, 0)
# plot_and_play(test_audio, 26, 5, 0)
# plot_and_play(test_audio, 28, 5, 0)
# plot_and_play(test_audio, 30, 5, 0)

# TODO: Check if all the audio files have the same number of channels

# TODO: Loop through all music file to get the max length spectrogram, and other specs
# Spectrogram length for 45s audio with freq 44100 is often 15523
# Largeest 3 spectrogram, 16874 at 1198.wav, 103922 at 2001.wav, 216080 at 2011.wav
# The reason why there are multiple spectrogram is because the music have different length
# For the exact 45 seconds audio, the spectrogram time length is 15502.

# SPECTROGRAM_TIME_LENGTH = 15502
# min_audio_length = 1e8
# for fname in os.listdir(AUDIO_FOLDER):
#   song_path = os.path.join(AUDIO_FOLDER, fname)
#   audio_file = tf.io.read_file(song_path)
#   waveforms, _ = tf.audio.decode_wav(contents=audio_file)
#   audio_length = waveforms.shape[0] // DEFAULT_FREQ
#   if audio_length < min_audio_length:
#     min_audio_length = audio_length
#     print(f"The min audio time length is: {min_audio_length} second(s) at {fname}")
#   spectrogram = get_spectrogram(waveforms[..., 0], input_len=waveforms.shape[0])
#   if spectrogram.shape[0] > SPECTROGRAM_TIME_LENGTH:
#     SPECTROGRAM_TIME_LENGTH = spectrogram.shape[0]
#     print(f"The max spectrogram time length is: {SPECTROGRAM_TIME_LENGTH} at {fname}")

# TODO: Get the max and min val of the label. Mean:

def train_datagen_song_level():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  while True:
    # Reset pointer
    if pointer >= len(train_df):
      pointer = 0

    row = train_df.loc[pointer]
    song_id = row["song_id"]
    valence_mean = row["valence_mean"]
    arousal_mean = row["arousal_mean"]
    label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
    song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)
    # print(waveforms.shape)

    # Work on building spectrogram
    # Shape (timestep, frequency, n_channel)
    spectrograms = None
    # Loop through each channel
    for i in range(waveforms.shape[-1]):
      # Shape (timestep, frequency, 1)
      spectrogram = get_spectrogram(waveforms[..., i], input_len=waveforms.shape[0])
      if spectrograms == None:
        spectrograms = spectrogram
      else:
        spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)
    pointer += 1

    padded_spectrogram = np.zeros((SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
    # spectrograms = spectrograms[tf.newaxis, ...]
    # some spectrogram are not the same shape
    padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
    
    yield (tf.convert_to_tensor(padded_spectrogram), label)

def test_datagen_song_level():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  while True:
    # Reset pointer
    if pointer >= len(test_df):
      pointer = 0

    row = test_df.loc[pointer]
    song_id = row["song_id"]
    valence_mean = row["valence_mean"]
    arousal_mean = row["arousal_mean"]
    label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
    song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)
    # print(waveforms.shape)

    # Work on building spectrogram
    # Shape (timestep, frequency, n_channel)
    spectrograms = None
    # Loop through each channel
    for i in range(waveforms.shape[-1]):
      # Shape (timestep, frequency, 1)
      spectrogram = get_spectrogram(waveforms[..., i], input_len=waveforms.shape[0])
      if spectrograms == None:
        spectrograms = spectrogram
      else:
        spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)
    pointer += 1

    padded_spectrogram = np.zeros((SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
    # spectrograms = spectrograms[tf.newaxis, ...]
    # some spectrogram are not the same shape
    padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
    
    yield (tf.convert_to_tensor(padded_spectrogram), label)

train_dataset = tf.data.Dataset.from_generator(
  train_datagen_song_level,
  output_signature=(
    tf.TensorSpec(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(2), dtype=tf.float32)
  )
)
train_batch_dataset = train_dataset.batch(BATCH_SIZE)
train_batch_iter = iter(train_batch_dataset)

test_dataset = tf.data.Dataset.from_generator(
  test_datagen_song_level,
  output_signature=(
    tf.TensorSpec(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(2, ), dtype=tf.float32)
  )
)
test_batch_dataset = test_dataset.batch(BATCH_SIZE)
test_batch_iter = iter(test_batch_dataset)

# %%

## Training

def train_step(batch_x, batch_label, model, loss_function, optimizer, step=-1):
  with tf.device("/GPU:0"):
    with tf.GradientTape() as tape:
      logits = model(batch_x, training=True)
      loss = loss_function(batch_label, logits)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
  return loss

def train(model, 
        training_batch_iter, 
        test_batch_iter, 
        optimizer, 
        loss_function,
        epochs=1, 
        steps_per_epoch=20, 
        valid_step=5,
        history_path=None,
        weights_path=None,
        save_history=False):
  
  if history_path != None and os.path.exists(history_path):
    # Sometimes, we have not created the files
    with open(history_path, "rb") as f:
      history = np.load(f, allow_pickle=True)
    epochs_loss, epochs_val_loss = history
    epochs_loss = epochs_loss.tolist()
    epochs_val_loss = epochs_val_loss.tolist()
  else:
    epochs_val_loss = []
    epochs_loss = []
  
  if weights_path != None and os.path.exists(weights_path + ".index"):

    model.load_weights(weights_path)
    print("Model weights loaded!")

  for epoch in range(epochs):
    losses = []

    with tf.device("/CPU:0"):
      step_pointer = 0
      while step_pointer < steps_per_epoch:
        batch = next(training_batch_iter)
        batch_x = batch[0]
        batch_label = batch[1]
        loss = train_step(batch_x, batch_label, model, loss_function, optimizer, step=step_pointer + 1)
        print(f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}")
        losses.append(loss)

        if (step_pointer + 1) % valid_step == 0:
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step_pointer + 1, float(loss))
          )
          # perform validation
          val_batch = next(test_batch_iter)
          logits = model(val_batch[0], training=False)
          val_loss = loss_function(val_batch[1], logits)
          print(f"Validation loss: {val_loss}\n-----------------")
        if (step_pointer + 1) == steps_per_epoch:
          val_batch = next(test_batch_iter)
          logits = model(val_batch[0], training=False)
          val_loss = loss_function(val_batch[1], logits)
          epochs_val_loss.append(val_loss)

        step_pointer += 1
    epochs_loss.append(losses)

    # Save history and model
    if history_path != None and save_history:
      np.save(history_path, [epochs_loss, epochs_val_loss])
    
    if weights_path != None:
      model.save_weights(weights_path)
  
  # return history
  return [epochs_loss, epochs_val_loss]

# %%

model = SimpleDenseModel(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL, BATCH_SIZE)
model.build(input_shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL))
model.model().summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  simple_mse_loss,
  epochs=10,
  steps_per_epoch=100, # 1800 // 16
  valid_step=25,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True
)


# %%

# Plot
with open(history_path, "rb") as f:
  [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)


e_loss = [k[0] for k in epochs_loss]

plt.plot(np.arange(1,len(e_loss)+ 1), e_loss, label = "train loss")
plt.plot(np.arange(1,len(epochs_val_loss)+ 1), epochs_val_loss, label = "val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%5

# model.load_weights(weights_path)
# model.trainable_weights
# y.shape


