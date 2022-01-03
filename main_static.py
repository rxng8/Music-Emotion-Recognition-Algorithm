"""
  file: main_static.py
  author: Alex Nguyen
  This file contains code to process the whole song labeled data (statically labeled)
"""
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
from mer.model import Simple_CRNN_3, SimpleDenseModel, \
  SimpleConvModel, \
  ConvBlock, \
  ConvBlock2,\
  Simple_CRNN, \
  Simple_CRNN_2, \
  Simple_CRNN_3

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
      # spectrogram = tf.convert_to_tensor(np.log(spectrogram.numpy() + np.finfo(float).eps))
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
      # spectrogram = tf.convert_to_tensor(np.log(spectrogram.numpy() + np.finfo(float).eps))
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
# train_batch_dataset = train_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
train_batch_iter = iter(train_batch_dataset)


# Comment out to decide to create a normalization layer.
# NOTE: this is every time consuming because it looks at all the data, only 
# use this at the first time.
# NOTE: Normally, we create this layer once, save it somewhere to reuse in
# every other model.
#
# norm_layer = L.Normalization()
# norm_layer.adapt(data=train_dataset.map(map_func=lambda spec, label: spec))
#

test_dataset = tf.data.Dataset.from_generator(
  test_datagen_song_level,
  output_signature=(
    tf.TensorSpec(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=tf.float32),
    tf.TensorSpec(shape=(2, ), dtype=tf.float32)
  )
)
test_batch_dataset = test_dataset.batch(BATCH_SIZE)
# test_batch_dataset = test_batch_dataset.cache().prefetch(tf.data.AUTOTUNE) # OOM error
test_batch_iter = iter(test_batch_dataset)

# ds = iter(train_dataset)
# i, o = next(ds)
# log_spec = np.log(i + np.finfo(float).eps)

# print(tf.reduce_max(i))
# print(tf.reduce_min(i))
# print(tf.reduce_mean(i))

# print(tf.reduce_max(log_spec))
# print(tf.reduce_min(log_spec))
# print(tf.reduce_mean(log_spec))

# ii = tf.transpose(i[..., 0], [1,0])
# height = ii.shape[0]
# width = ii.shape[1]
# X = np.linspace(0, np.size(ii), num=width, dtype=int)
# Y = range(height)
# plt.pcolormesh(X, Y, ii)
# plt.show()


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

## Define model first

weights_path = "./weights/cbam_2/checkpoint"
history_path = "./history/cbam_2.npy"

# model = SimpleDenseModel(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL, BATCH_SIZE)
# model.build(input_shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL))
# model.model().summary()

# model = SimpleConvModel(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL, BATCH_SIZE)
# model.model.load_weights(weights_path)

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

# %%

model = Simple_CRNN_3()
# model.summary()
sample_input = tf.ones(shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
with tf.device("/CPU:0"):
  sample_output = model(sample_input, training=False)
print(sample_output)

# %%

# About 50 epochs with each epoch step 100 will cover the whole training dataset!
history = train(
  model,
  train_batch_iter,
  test_batch_iter,
  optimizer,
  simple_mae_loss,
  epochs=2,
  steps_per_epoch=100, # 1800 // 16
  valid_step=20,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True
)


# %%

### MODEL DEBUGGING ###

def base_cnn():
  """ Base CNN Feature extractor for 45 second spectrogram
    Input to model shape: (SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2)
    Output of the model shape: (4, 60, 256) 
      (Convolved frequency, convolved timestep, feature neurons)
  Returns:
    tf.keras.Model: Return a model
  """

  inputs = L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
  
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = L.Resizing(FREQUENCY_LENGTH, 1024)(tensor)
  
  tensor = L.Conv2D(64, (5,5), padding="valid", name="conv_1_1")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(64 // 2, (1,1), padding="valid", name="conv_1_2")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(128, (5,5), padding="valid", name="conv_2_1")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(128 // 2, (1,1), padding="valid", name="conv_2_2")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(256, (5,5), padding="valid", name="conv_3_1")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(256 // 2, (1,1), padding="valid", name="conv_3_2")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(512, (5,5), padding="valid", name="conv_4_1")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(512 // 2, (1,1), padding="valid", name="conv_4_2")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  out = L.Dropout(0.1)(tensor)
  model = tf.keras.Model(inputs=inputs, outputs=out, name="base_model")
  return model

class ChannelAttention(tf.keras.layers.Layer):
  def __init__(self, neuron: int, ratio: int, use_average=True, **kwargs) -> None:
    super().__init__(**kwargs)
    self.neuron = neuron
    self.ratio = ratio
    self.use_average = use_average

  def build(self, input_shape):
    """build layers

    Args:
      input_shape (tf.shape): the shape of the input

    Returns:
      [type]: [description]
    """
    assert len(input_shape) == 4, "The input shape to the layer has to be 3D"
    self.first_shared_layer = L.Dense(self.neuron // self.ratio, activation="relu", kernel_initializer="he_normal")
    self.second_shared_layer = L.Dense(self.neuron, activation="relu", kernel_initializer="he_normal")

  def call(self, inputs):
    if self.use_average:
      avg_pool_tensor = L.GlobalAveragePooling2D()(inputs) # Shape (batch, filters)
      avg_pool_tensor = L.Reshape((1,1,-1))(avg_pool_tensor) # Shape (batch, 1, 1, filters)
      avg_pool_tensor = self.first_shared_layer(avg_pool_tensor)
      avg_pool_tensor = self.second_shared_layer(avg_pool_tensor)

      max_pool_tensor = L.GlobalMaxPool2D()(inputs) # Shape (batch, filters)
      max_pool_tensor = L.Reshape((1,1,-1))(max_pool_tensor) # Shape (batch, 1, 1, filters)
      max_pool_tensor = self.first_shared_layer(max_pool_tensor)
      max_pool_tensor = self.second_shared_layer(max_pool_tensor)

      attention_tensor = L.Add()([avg_pool_tensor, max_pool_tensor])
      attention_tensor = L.Activation("sigmoid")(attention_tensor)

      out = L.Multiply()([inputs, attention_tensor]) # Broadcast element-wise multiply. (batch, height, width, filters) x (batch, 1, 1, neurons) 

      return out
    else:
      max_pool_tensor = L.GlobalMaxPool2D()(inputs) # Shape (batch, filters)
      max_pool_tensor = L.Reshape((1,1,-1))(max_pool_tensor) # Shape (batch, 1, 1, filters)
      max_pool_tensor = self.first_shared_layer(max_pool_tensor)
      max_pool_tensor = self.second_shared_layer(max_pool_tensor)
      attention_tensor = L.Activation("sigmoid")(max_pool_tensor)
      out = L.Multiply()([inputs, attention_tensor])
      return out

class SpatialAttention(tf.keras.layers.Layer):
  def __init__(self, kernel_size, use_average=True, **kwargs) -> None:
    super().__init__(**kwargs)
    self.kernel_size = kernel_size
    self.use_average = use_average

  def build(self, input_shape):
    """build layers

    Args:
      input_shape (tf.shape): the shape of the input

    Returns:
      [type]: [description]
    """
    assert len(input_shape) == 4, "The input shape to the layer has to be 3D"
    self.conv_layer = L.Conv2D(1, self.kernel_size, padding="same", activation="relu",
      kernel_initializer="he_normal")

  def call(self, inputs):
    if self.use_average:
      avg_pool_tensor = L.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(inputs)
      max_pool_tensor = L.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
      concat_tensor = L.Concatenate(axis=-1)([avg_pool_tensor, max_pool_tensor])
      tensor = self.conv_layer(concat_tensor) # shape (height, width, 1)
      out = L.Multiply()([inputs, tensor]) # Broadcast element-wise multiply. (batch, height, width, neurons) x (batch, height, width, 1) 

      return out
    else:
      max_pool_tensor = L.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(inputs)
      tensor = self.conv_layer(max_pool_tensor) # shape (height, width, 1)
      out = L.Multiply()([inputs, tensor]) # Broadcast element-wise multiply. (batch, height, width, neurons) x (batch, height, width, 1) 

      return out

class CBAM_Block(tf.keras.layers.Layer):
  """ TODO: Implement Res Block architecture for CBAM Block

  Args:
      tf ([type]): [description]
  """
  def __init__(self,
      channel_attention_filters, 
      channel_attention_ratio, 
      spatial_attention_kernel_size, 
      **kwargs) -> None:
    super().__init__(**kwargs)
    self.channel_attention_filters = channel_attention_filters
    self.channel_attention_ratio = channel_attention_ratio
    self.spatial_attention_kernel_size = spatial_attention_kernel_size
  
  def build(self, input_shape):
    assert len(input_shape) == 4, "The shape must be 3D!"

    # NOTE: The reson why self.channel_attention_filters is put here is because the number
    # of neurons of the input to channel attention has to be equal the number of filters
    # in the channel attention
    self.conv_1 = L.Conv2D(self.channel_attention_filters * 2, (5,5), padding="same", activation="relu")
    self.conv_2 = L.Conv2D(self.channel_attention_filters, (1,1), padding="same", activation="relu")
    self.c_att = ChannelAttention(self.channel_attention_filters, self.channel_attention_ratio)
    self.s_att = SpatialAttention(self.spatial_attention_kernel_size)

  def call(self, inputs):
    # inputs shape (batch, height, width, channel)
    tensor = self.conv_1(inputs) # shape (batch, height, width, filters * 2)
    tensor = self.conv_2(tensor) # shape (batch, height, width, filters)
    tensor = self.c_att(tensor) # shape (batch, height, width, filters)
    tensor = self.s_att(tensor) # shape (batch, height, width, filters)
    return tensor

def cbam_1():

  inputs = L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = L.Resizing(FREQUENCY_LENGTH, 1024)(tensor)

  # tensor = CBAM_Block(32, 2, (5,5))(tensor)
  tensor = L.Conv2D(64, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(32, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_1 = ChannelAttention(32, 2)(tensor)
  tensor_att_1 = SpatialAttention((5,5))(tensor_att_1)
  tensor = L.Add()([tensor, tensor_att_1])
  tensor = L.MaxPool2D(2,2)(tensor)

  # tensor = CBAM_Block(64, 2, (7,7))(tensor)
  tensor = L.Conv2D(128, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(64, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_2 = ChannelAttention(64, 2)(tensor)
  tensor_att_2 = SpatialAttention((7,7))(tensor_att_2)
  tensor = L.Add()([tensor, tensor_att_2])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)

  # tensor = CBAM_Block(128, 2, (7,7))(tensor)
  tensor = L.Conv2D(256, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(128, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_3 = ChannelAttention(128, 2)(tensor)
  tensor_att_3 = SpatialAttention((7,7))(tensor_att_3)
  tensor = L.Add()([tensor, tensor_att_3])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)
  
  # tensor = CBAM_Block(256, 2, (5,5))(tensor)
  tensor = L.Conv2D(512, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(256, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_4 = ChannelAttention(256, 2)(tensor)
  tensor_att_4 = SpatialAttention((5,5))(tensor_att_4)
  tensor = L.Add()([tensor, tensor_att_4])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)

  # tensor = CBAM_Block(256, 2, (3,3))(tensor)
  tensor = L.Conv2D(512, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(256, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_5 = ChannelAttention(256, 2)(tensor)
  tensor_att_5 = SpatialAttention((3,3))(tensor_att_5)
  tensor = L.Add()([tensor, tensor_att_5])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)

  tensor = L.Permute((2, 1, 3))(tensor)
  tensor = L.Reshape((32, 4 * 256))(tensor)

  # tensor = L.GRU(256, activation="tanh", return_sequences=True)(tensor)
  # tensor = L.GRU(128, activation="tanh", return_sequences=True)(tensor)
  tensor = L.GRU(64, activation="tanh")(tensor)
  tensor = L.Dense(512, activation="relu")(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  out = L.Dense(2, activation="relu")(tensor)

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model

def cbam_2():
  """ No average CBAM

  Returns:
    tf.keras.Model: The Model
  """
  inputs = L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = L.Resizing(FREQUENCY_LENGTH, 1024)(tensor)

  # tensor = CBAM_Block(32, 2, (5,5))(tensor)
  tensor = L.Conv2D(64, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(32, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_1 = ChannelAttention(32, 2, use_average=False)(tensor)
  tensor_att_1 = SpatialAttention((5,5), use_average=False)(tensor_att_1)
  tensor = L.Add()([tensor, tensor_att_1])
  tensor = L.MaxPool2D(2,2)(tensor)

  # tensor = CBAM_Block(64, 2, (7,7))(tensor)
  tensor = L.Conv2D(128, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(64, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_2 = ChannelAttention(64, 2, use_average=False)(tensor)
  tensor_att_2 = SpatialAttention((7,7), use_average=False)(tensor_att_2)
  tensor = L.Add()([tensor, tensor_att_2])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)

  # tensor = CBAM_Block(128, 2, (7,7))(tensor)
  tensor = L.Conv2D(256, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(128, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_3 = ChannelAttention(128, 2, use_average=False)(tensor)
  tensor_att_3 = SpatialAttention((7,7), use_average=False)(tensor_att_3)
  tensor = L.Add()([tensor, tensor_att_3])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)
  
  # tensor = CBAM_Block(256, 2, (5,5))(tensor)
  tensor = L.Conv2D(512, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(256, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_4 = ChannelAttention(256, 2, use_average=False)(tensor)
  tensor_att_4 = SpatialAttention((5,5), use_average=False)(tensor_att_4)
  tensor = L.Add()([tensor, tensor_att_4])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)

  # tensor = CBAM_Block(256, 2, (3,3))(tensor)
  tensor = L.Conv2D(512, (5,5), padding="same", activation="relu")(tensor)
  tensor = L.Conv2D(256, (1,1), padding="same", activation="relu")(tensor)
  tensor_att_5 = ChannelAttention(256, 2, use_average=False)(tensor)
  tensor_att_5 = SpatialAttention((3,3), use_average=False)(tensor_att_5)
  tensor = L.Add()([tensor, tensor_att_5])
  tensor = L.MaxPool2D(2,2)(tensor)
  # tensor = L.BatchNormalization()(tensor)
  # tensor = L.Dropout(0.1)(tensor)

  tensor = L.Permute((2, 1, 3))(tensor)
  tensor = L.Reshape((32, 4 * 256))(tensor)

  # tensor = L.GRU(256, activation="tanh", return_sequences=True)(tensor)
  # tensor = L.GRU(128, activation="tanh", return_sequences=True)(tensor)
  tensor = L.LSTM(256, activation="tanh")(tensor)
  tensor = L.Dense(512, activation="relu")(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  out = L.Dense(2, activation="relu")(tensor)

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model

model = cbam_2()
model.summary()
sample_input = tf.ones(shape=(BATCH_SIZE, SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
with tf.device("/CPU:0"):
  sample_output = model(sample_input, training=False)
print(sample_output.shape)

# TODO: Code the CBAM architecture
# TODO: Code the attetion after the CBAM



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
evaluate(i, model, simple_mae_loss, play=False)

# %%

####### INTERMEDIARY REPRESENTATION ########

layer_list = [l for l in model.layers]
debugging_model = tf.keras.Model(inputs=model.inputs, outputs=[l.output for l in layer_list])

# %%

layer_list

# %%

test_id = 223
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

f, axarr = plt.subplots(8,8, figsize=(25,15))
CONVOLUTION_NUMBER_LIST = [8, 9, 10, 11, 12, 13, 14, 15]
LAYER_LIST = [10, 11, 12, 13, 16, 17, 18, 19]
for x, CONVOLUTION_NUMBER in enumerate(CONVOLUTION_NUMBER_LIST):
  f1 = y_pred_list[LAYER_LIST[0]]
  plot_spectrogram(tf.transpose(f1[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[0,x])
  axarr[0,x].grid(False)
  f2 = y_pred_list[LAYER_LIST[1]]
  plot_spectrogram(tf.transpose(f2[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[1,x])
  axarr[1,x].grid(False)
  f3 = y_pred_list[LAYER_LIST[2]]
  plot_spectrogram(tf.transpose(f3[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[2,x])
  axarr[2,x].grid(False)
  f4 = y_pred_list[LAYER_LIST[3]]
  plot_spectrogram(tf.transpose(f4[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[3,x])
  axarr[3,x].grid(False)
  
  f5 = y_pred_list[LAYER_LIST[4]]
  plot_spectrogram(tf.transpose(f5[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[4,x])
  axarr[4,x].grid(False)
  f6 = y_pred_list[LAYER_LIST[5]]
  plot_spectrogram(tf.transpose(f6[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[5,x])
  axarr[5,x].grid(False)
  f7 = y_pred_list[LAYER_LIST[6]]
  plot_spectrogram(tf.transpose(f7[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[6,x])
  axarr[6,x].grid(False)
  f8 = y_pred_list[LAYER_LIST[7]]
  plot_spectrogram(tf.transpose(f8[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[7,x])
  axarr[7,x].grid(False)

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


f4 = y_pred_list[3]

# %%

f4.shape

# %%

f4

# %%

w = layer_list[5].weights[0]
w
# %%

y_pred_list[-1]

# %%

