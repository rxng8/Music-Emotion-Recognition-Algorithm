# %%

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import sounddevice as sd
import pandas as pd
import tensorflow.keras.layers as L

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from tensorflow.python.keras.layers.core import Dropout

from mer.utils import get_spectrogram, \
  plot_spectrogram, \
  load_metadata, \
  plot_and_play, \
  preprocess_waveforms, \
  split_train_test

from mer.const import *
from mer.loss import simple_mse_loss, simple_mae_loss
from mer.model import SimpleDenseModel, \
  SimpleConvModel, \
  ConvBlock, \
  ConvBlock2,\
  Simple_CRNN, \
  Simple_CRNN_2

# Set the seed value for experiment reproducibility.
# seed = 42
# tf.random.set_seed(seed)
# np.random.seed(seed)

sd.default.samplerate = DEFAULT_FREQ

ANNOTATION_SONG_LEVEL = "./dataset/DEAM/annotations/annotations averaged per song/song_level/"
AUDIO_FOLDER = "./dataset/DEAM/wav"
filenames = tf.io.gfile.glob(str(AUDIO_FOLDER) + '/*')

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
    valence_mean = float(row["valence_mean"])
    arousal_mean = float(row["arousal_mean"])
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
    valence_mean = float(row["valence_mean"])
    arousal_mean = float(row["arousal_mean"])
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
    try:
      model.load_weights(weights_path)
      print("Model weights loaded!")
    except:
      print("cannot load weights!")

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
          print(f"exmaple logits: {logits}")
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

"""################## Training #################"""

weights_path = "./weights/simple_crnn_2/checkpoint"
history_path = "./history/simple_crnn_2.npy"

# model = SimpleDenseModel(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL, BATCH_SIZE)
# model.build(input_shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL))
# model.model().summary()

# model = SimpleConvModel(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL, BATCH_SIZE)
# model.model.load_weights(weights_path)

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  simple_mae_loss,
  epochs=10,
  steps_per_epoch=100, # 1800 // 16
  valid_step=20,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True
)

# %%

### DEBUG ### 

model = Simple_CRNN_2()
# model.summary()
sample_input = tf.ones(shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
with tf.device("/CPU:0"):
  sample_output = model(sample_input, training=False)
print(sample_output)

# %%




# %%5


sample_output.shape

# %%








# %%

# Plot
with open(history_path, "rb") as f:
  [epochs_loss, epochs_val_loss] = np.load(f, allow_pickle=True)


e_loss = [k[0] for k in epochs_loss]

e_all_loss = []

id = 0
time_val = []
for epoch in epochs_loss:
  for step in epoch:
    e_all_loss.append(step.numpy())
    id += 1
  time_val.append(id)

# %%

plt.plot(np.arange(0, len(e_all_loss), 1), e_all_loss, label = "train loss")
plt.plot(time_val, epochs_val_loss, label = "val loss")

# plt.plot(np.arange(1,len(e_loss)+ 1), e_loss, label = "train loss")
# plt.plot(np.arange(1,len(epochs_val_loss)+ 1), epochs_val_loss, label = "val loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%5

# model.load_weights(weights_path)
# model.trainable_weights
# y.shape


# %%



# model.build(input_shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL))
# model.summary()
# %%

model.save_weights(weights_path)


# %%



model.load_weights(weights_path)


# %%


def evaluate(df_pointer, model, loss_func, play=False):
  row = test_df.loc[df_pointer]
  song_id = row["song_id"]
  valence_mean = row["valence_mean"]
  arousal_mean = row["arousal_mean"]
  label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
  print(f"Label: Valence: {valence_mean}, Arousal: {arousal_mean}")
  song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
  audio_file = tf.io.read_file(song_path)
  waveforms, _ = tf.audio.decode_wav(contents=audio_file)
  waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)
  spectrograms = None
  # Loop through each channel
  for i in range(waveforms.shape[-1]):
    # Shape (timestep, frequency, 1)
    spectrogram = get_spectrogram(waveforms[..., i], input_len=waveforms.shape[0])
    if spectrograms == None:
      spectrograms = spectrogram
    else:
      spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)

  spectrograms = spectrograms[tf.newaxis, ...]

  ## Eval
  y_pred = model(spectrograms, training=False)[0]
  print(f"Predicted y_pred value: Valence: {y_pred[0]}, Arousal: {y_pred[1]}")

  loss = loss_func(label[tf.newaxis, ...], y_pred)
  print(f"Loss: {loss}")

  if play:
    plot_and_play(waveforms, 0, 40, 0)

i = 0

# %%

i += 1
evaluate(i, model, simple_mae_loss, play=True)

# %%

####### INTERMEDIARY REPRESENTATION ########

layer_list = [l for l in model.layers]
debugging_model = tf.keras.Model(inputs=model.inputs, outputs=[l.output for l in layer_list])

# %%

layer_list

# %%

test_id = 234
row = test_df.loc[test_id]
song_id = row["song_id"]
valence_mean = row["valence_mean"]
arousal_mean = row["arousal_mean"]
label = tf.convert_to_tensor([valence_mean, arousal_mean], dtype=tf.float32)
print(f"Label: Valence: {valence_mean}, Arousal: {arousal_mean}")
song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
audio_file = tf.io.read_file(song_path)
waveforms, _ = tf.audio.decode_wav(contents=audio_file)
waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)
spectrograms = None
# Loop through each channel
for i in range(waveforms.shape[-1]):
  # Shape (timestep, frequency, 1)
  spectrogram = get_spectrogram(waveforms[..., i], input_len=waveforms.shape[0])
  if spectrograms == None:
    spectrograms = spectrogram
  else:
    spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)

spectrograms = spectrograms[tf.newaxis, ...]

print(label)
# plot_and_play(waveforms, 0, 40, 0)

## Eval
y_pred_list = debugging_model(spectrograms, training=False)
print(f"Predicted y_pred value: Valence: {y_pred_list[-1][0, 0]}, Arousal: {y_pred_list[-1][0, 1]}")


# %%

def show_color_mesh(spectrogram):
  """ Generate color mesh

  Args:
      spectrogram (2D array): Expect shape (Frequency length, time step)
  """
  assert len(spectrogram.shape) == 2
  log_spec = np.log(spectrogram + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  plt.pcolormesh(X, Y, log_spec)
  plt.show()

show_color_mesh(tf.transpose(spectrograms[0, :, :, 0], [1,0]))


# %%

f, axarr = plt.subplots(4,4, figsize=(25,15))
CONVOLUTION_NUMBER_LIST = [0, 4, 7, 23]
for x, CONVOLUTION_NUMBER in enumerate(CONVOLUTION_NUMBER_LIST):
  f1 = y_pred_list[4]
  plot_spectrogram(tf.transpose(f1[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[0,x])
  axarr[0,x].grid(False)
  f2 = y_pred_list[6]
  plot_spectrogram(tf.transpose(f2[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[1,x])
  axarr[1,x].grid(False)
  f3 = y_pred_list[10]
  plot_spectrogram(tf.transpose(f3[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[2,x])
  axarr[2,x].grid(False)
  f4 = y_pred_list[22]
  plot_spectrogram(tf.transpose(f4[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[3,x])
  axarr[3,x].grid(False)

axarr[0,0].set_ylabel("After convolution layer 1")
axarr[1,0].set_ylabel("After convolution layer 2")
axarr[2,0].set_ylabel("After convolution layer 3")
axarr[3,0].set_ylabel("After convolution layer 7")

axarr[0,0].set_title("convolution number 0")
axarr[0,1].set_title("convolution number 4")
axarr[0,2].set_title("convolution number 7")
axarr[0,3].set_title("convolution number 23")

plt.show()

# %%


f4 = y_pred_list[21]

# %%

f4.shape

# %%

f4

# %%

layer_list[-1].weights[0]

# %%

y_pred_list[-1]

# %%

