"""
  file: main_dynamic.py
  author: Alex Nguyen
  This file contains code to process the each-second song labeled data (dynamically labeled)
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
  split_train_test, \
  tanh_to_sigmoid

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

ANNOTATION_SONG_LEVEL = "./dataset/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/"
AUDIO_FOLDER = "./dataset/DEAM/wav"
filenames = tf.io.gfile.glob(str(AUDIO_FOLDER) + '/*')

BATCH_SIZE = 8
DEFAULT_SECOND_PER_TIME_STEP = 0.5

# Process with average annotation per song. 
df = load_metadata(ANNOTATION_SONG_LEVEL)

train_df, test_df = split_train_test(df, TRAIN_RATIO)
# Process with average annotation per second. 
valence_df = pd.read_csv(os.path.join(ANNOTATION_SONG_LEVEL, "valence.csv"), sep=r"\s*,\s*", engine="python")
arousal_df = pd.read_csv(os.path.join(ANNOTATION_SONG_LEVEL, "arousal.csv"), sep=r"\s*,\s*", engine="python")

assert len(valence_df) == len(arousal_df)

song_id_df = pd.DataFrame({"song_id": valence_df["song_id"]})

# NOTE:  Split only the song id table, then join that table with valence_df and arousal_df
train_song_ids, test_song_ids = split_train_test(song_id_df, TRAIN_RATIO)

train_valence_df = train_song_ids.merge(valence_df, on="song_id", how="left").dropna(axis=1)
train_arousal_df = train_song_ids.merge(arousal_df, on="song_id", how="left").dropna(axis=1)

test_valence_df = test_song_ids.merge(valence_df, on="song_id", how="left").dropna(axis=1)
test_arousal_df = test_song_ids.merge(arousal_df, on="song_id", how="left").dropna(axis=1)

# Describe (Summaraize) the datasets label
# train_valence_flatten_df = pd.DataFrame(np.reshape(train_valence_df.loc[:, train_valence_df.columns != "song_id"].to_numpy(), (-1,)))
# print(train_valence_flatten_df.describe())

# test_valence_flatten_df = pd.DataFrame(np.reshape(test_valence_df.loc[:, test_valence_df.columns != "song_id"].to_numpy(), (-1,)))
# print(test_valence_flatten_df.describe())

# Debugging
# pointer = 0
# row = train_valence_df.loc[pointer]
# song_id = row["song_id"]
# assert train_arousal_df.loc[pointer, "song_id"] == song_id, "Wrong row!"
# # Load song and waveform
# song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
# audio_file = tf.io.read_file(song_path)
# waveforms, _ = tf.audio.decode_wav(contents=audio_file)
# # Pad to max 45 second. Shape (total_frequency, n_channels)
# waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)

# # Get the labels series
# valence_labels = train_valence_df.loc[pointer, train_valence_df.columns != "song_id"]
# arousal_labels = train_arousal_df.loc[pointer, train_arousal_df.columns != "song_id"]

# time_pointer = 8
# time_end_point = MIN_TIME_END_POINT + time_pointer * DEFAULT_SECOND_PER_TIME_STEP # 15 + ptr * 0.5

# end_wave_index = int(time_end_point * DEFAULT_FREQ)
# start_wave_index = int(end_wave_index - WINDOW_SIZE)

# current_waveforms = waveforms[start_wave_index: end_wave_index, ...]
# # Work on building spectrogram
# # Shape (timestep, frequency, n_channel)
# spectrograms = None
# # Loop through each channel

# test_wave = get_spectrogram(current_waveforms[..., 0], input_len=current_waveforms.shape[0])
# print(test_wave.shape) # TensorShape([171, 129, 1])

# for i in range(current_waveforms.shape[-1]):
#   # Shape (timestep, frequency, 1)
#   spectrogram = get_spectrogram(current_waveforms[..., i], input_len=current_waveforms.shape[0])
#   # spectrogram = tf.convert_to_tensor(np.log(spectrogram.numpy() + np.finfo(float).eps))
#   if spectrograms == None:
#     spectrograms = spectrogram
#   else:
#     spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)
# pointer += 1

# padded_spectrogram = np.zeros((SPECTROGRAM_HALF_SECOND_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
# # spectrograms = spectrograms[tf.newaxis, ...]
# # some spectrogram are not the same shape
# padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms

# print(padded_spectrogram.shape)

def train_datagen_per_second():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  while True:
    # Reset pointer
    if pointer >= len(train_valence_df):
      pointer = 0

    row = train_valence_df.loc[pointer]
    song_id = row["song_id"]
    assert train_arousal_df.loc[pointer, "song_id"] == song_id, "Wrong row!"
    
    # Load song and waveform
    song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    # Pad to max 45 second. Shape (total_frequency, n_channels)
    waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)

    # Get the labels series
    valence_labels = train_valence_df.loc[pointer, train_valence_df.columns != "song_id"]
    arousal_labels = train_arousal_df.loc[pointer, train_arousal_df.columns != "song_id"]
    # Loop through the series
    for time_pointer, ((valence_time_name, valence), (arousal_time_name, arousal)) in enumerate(zip(valence_labels.iteritems(), arousal_labels.iteritems())):
      label = tf.convert_to_tensor([tanh_to_sigmoid(valence), tanh_to_sigmoid(arousal)], dtype=tf.float32)
      time_end_point = MIN_TIME_END_POINT + time_pointer * DEFAULT_SECOND_PER_TIME_STEP # 15 + ptr * 0.5

      end_wave_index = int(time_end_point * DEFAULT_FREQ)
      start_wave_index = int(end_wave_index - WINDOW_SIZE)
      
      try:
        current_waveforms = waveforms[start_wave_index: end_wave_index, ...]
        # Work on building spectrogram
        # Shape (timestep, frequency, n_channel)
        spectrograms = None
        # Loop through each channel
        for i in range(current_waveforms.shape[-1]):
          # Shape (timestep, frequency, 1)
          spectrogram = get_spectrogram(current_waveforms[..., i], input_len=current_waveforms.shape[0])
          # spectrogram = tf.convert_to_tensor(np.log(spectrogram.numpy() + np.finfo(float).eps))
          if spectrograms == None:
            spectrograms = spectrogram
          else:
            spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)
        pointer += 1

        padded_spectrogram = np.zeros((SPECTROGRAM_5_SECOND_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
        # spectrograms = spectrograms[tf.newaxis, ...]
        # some spectrogram are not the same shape
        padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
      
        yield (tf.convert_to_tensor(padded_spectrogram), label)
      
      except:
        print("There is some error accessing the waveforms by index")
        break

# train_dataset = tf.data.Dataset.from_generator(
#   train_datagen_per_second,
#   output_signature=(
#     tf.TensorSpec(shape=(SPECTROGRAM_HALF_SECOND_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=tf.float32),
#     tf.TensorSpec(shape=(2), dtype=tf.float32)
#   )
# )
# train_iter = iter(train_dataset)

def test_datagen_per_second():
  """ Predicting valence mean and arousal mean
  """
  pointer = 0
  while True:
    # Reset pointer
    if pointer >= len(test_valence_df):
      pointer = 0

    row = test_valence_df.loc[pointer]
    song_id = row["song_id"]
    assert test_arousal_df.loc[pointer, "song_id"] == song_id, "Wrong row!"
    
    # Load song and waveform
    song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
    audio_file = tf.io.read_file(song_path)
    waveforms, _ = tf.audio.decode_wav(contents=audio_file)
    # Pad to max 45 second. Shape (total_frequency, n_channels)
    waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)

    # Get the labels series
    valence_labels = test_valence_df.loc[pointer, test_valence_df.columns != "song_id"]
    arousal_labels = test_arousal_df.loc[pointer, test_arousal_df.columns != "song_id"]
    # Loop through the series
    for time_pointer, ((valence_time_name, valence), (arousal_time_name, arousal)) in enumerate(zip(valence_labels.iteritems(), arousal_labels.iteritems())):
      label = tf.convert_to_tensor([tanh_to_sigmoid(valence), tanh_to_sigmoid(arousal)], dtype=tf.float32)
      time_end_point = MIN_TIME_END_POINT + time_pointer * DEFAULT_SECOND_PER_TIME_STEP # 15 + ptr * 0.5

      end_wave_index = int(time_end_point * DEFAULT_FREQ)
      start_wave_index = int(end_wave_index - WINDOW_SIZE)
      
      try:
        current_waveforms = waveforms[start_wave_index: end_wave_index, ...]
        # Work on building spectrogram
        # Shape (timestep, frequency, n_channel)
        spectrograms = None
        # Loop through each channel
        for i in range(current_waveforms.shape[-1]):
          # Shape (timestep, frequency, 1)
          spectrogram = get_spectrogram(current_waveforms[..., i], input_len=current_waveforms.shape[0])
          # spectrogram = tf.convert_to_tensor(np.log(spectrogram.numpy() + np.finfo(float).eps))
          if spectrograms == None:
            spectrograms = spectrogram
          else:
            spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)
        pointer += 1

        padded_spectrogram = np.zeros((SPECTROGRAM_5_SECOND_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
        # spectrograms = spectrograms[tf.newaxis, ...]
        # some spectrogram are not the same shape
        padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
      
        yield (tf.convert_to_tensor(padded_spectrogram), label)
      
      except:
        print("There is some error accessing the waveforms by index")
        break


train_dataset = tf.data.Dataset.from_generator(
  train_datagen_per_second,
  output_signature=(
    tf.TensorSpec(shape=(SPECTROGRAM_5_SECOND_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=tf.float32),
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
  test_datagen_per_second,
  output_signature=(
    tf.TensorSpec(shape=(SPECTROGRAM_5_SECOND_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=tf.float32),
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

# it = iter(train_dataset)
# i, o = next(it)
# o.shape


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
    val_losses = []
    with tf.device("/CPU:0"):
      step_pointer = 0
      while step_pointer < steps_per_epoch:
        batch = next(training_batch_iter)
        batch_x = batch[0]
        batch_label = batch[1]
        loss = train_step(batch_x, batch_label, model, loss_function, optimizer, step=step_pointer + 1)
        print(f"Epoch {epoch + 1} - Step {step_pointer + 1} - Loss: {loss}")
        losses.append(loss)

        val_batch = next(test_batch_iter)
        logits = model(val_batch[0], training=False)
        val_loss = loss_function(val_batch[1], logits)
        val_losses.append(val_loss)

        if (step_pointer + 1) % valid_step == 0:
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step_pointer + 1, float(loss))
          )
          # perform validation
          print(f"exmaple logits: {logits}")
          print(f"Validation loss: {val_loss}\n-----------------")

        step_pointer += 1
    epochs_loss.append(losses)
    epochs_val_loss.append(val_losses)

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

weights_path = "./weights/dynamics/base_shallow_lstm/checkpoint"
history_path = "./history/dynamics/base_shallow_lstm.npy"

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
  steps_per_epoch=100, # 1200 // 16
  valid_step=20,
  history_path=history_path,
  weights_path=weights_path,
  save_history=True
)


# %%

### MODEL DEBUGGING ###

class ResBlock(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size_1=(5,5), kernel_size_2=(3,3), **kwargs) -> None:
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size_1 = kernel_size_1
    self.kernel_size_2 = kernel_size_2
  
  def build(self, input_shape):
    self.conv_norm = L.Conv2D(self.filters, (1, 1), padding="same")
    self.conv1 = L.Conv2D(self.filters, self.kernel_size_1, padding="same")
    self.conv2 = L.Conv2D(self.filters, self.kernel_size_2, padding="same")

  def call(self, inputs):
    skip_tensor = self.conv_norm(inputs)
    tensor = L.ReLU()(skip_tensor)
    tensor = self.conv1(tensor)
    tensor = L.ReLU()(tensor)
    tensor = self.conv2(tensor)
    tensor = L.Add()([skip_tensor, tensor])
    out = L.ReLU()(tensor)
    return out

class ResBlock2(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size=(5,5), **kwargs) -> None:
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
  
  def build(self, input_shape):
    self.conv_norm = L.Conv2D(self.filters, (1, 1), padding="same")
    self.conv1 = L.Conv2D(self.filters // 2, (1, 1), padding="same")
    self.conv2 = L.Conv2D(self.filters, self.kernel_size, padding="same")

  def call(self, inputs):
    skip_tensor = self.conv_norm(inputs)
    tensor = L.ReLU()(skip_tensor)
    tensor = self.conv1(tensor)
    tensor = L.ReLU()(tensor)
    tensor = self.conv2(tensor)
    tensor = L.Add()([skip_tensor, tensor])
    out = L.ReLU()(tensor)
    return out

def base_cnn(task_type="static"):
  """ Base CNN Feature extractor for 45 second spectrogram
    Input to model shape: (SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2)
    Output of the model shape: (4, 60, 256) 
      (Convolved frequency, convolved timestep, feature neurons)
  Args:
    task_type (str, optional): There are three value: 
      "static" for model evaluation per song, 
      "dynamic" for model evaluation per timestep and it takes the input waveform at only per timestep,
      "seq2seq" for model evaluation per second but take the input as the whole song. Defaults to "static".
  Returns:
    tf.keras.Model: Return a model
  """
  model = tf.keras.Sequential(name="base_cnn")
  if task_type == "static" or task_type == "seq2seq":
    model.add(L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2)))
    model.add(L.Permute((2, 1, 3)))
    model.add(L.Resizing(FREQUENCY_LENGTH, 1024))
  elif task_type == "dynamic":
    model.add(L.Input(shape=(SPECTROGRAM_5_SECOND_LENGTH, FREQUENCY_LENGTH, 2)))
    model.add(L.Permute((2, 1, 3)))
    model.add(L.Resizing(FREQUENCY_LENGTH, 1024))
  else:
    print("Wrong parameters")
    return

  cnn_config = [
    # [32, (5,5), (3, 3)],
    [64, (5,5), (3, 3)],
    [128, (5,5), (3, 3)],
    [256, (5,5), (3, 3)]
    # [512, (5,5), (3, 3)],
  ]

  for i, (filters, kernel_size_1, kernel_size_2) in enumerate(cnn_config):
    # model.add(ResBlock(filters, kernel_size_1, kernel_size_2, name=f"res_block_{i}"))
    # model.add(L.MaxPool2D(2,2, name=f"max_pool_{i}"))
    model.add(ResBlock(filters, kernel_size_1, name=f"res_block_{i}"))
    model.add(L.MaxPool2D(2,2, name=f"max_pool_{i}"))
  
  model.add(L.Conv2D(128, (1,1), activation="relu"))
  model.add(L.Conv2D(64, (1,1), activation="relu"))
  model.add(L.Conv2D(32, (1,1), activation="relu"))

  return model

def model():
  base = base_cnn(task_type="dynamic")
  
  tensor = base.outputs[0]
  tensor = L.Permute((2, 3, 1))(tensor)
  tensor = L.Reshape((128, -1))(tensor)
  tensor = L.LSTM(256)(tensor)
  tensor = L.Dense(256, activation=None)(tensor)
  tensor = L.Dense(64, activation=None)(tensor)
  out = L.Dense(2, activation=None)(tensor)

  model = tf.keras.Model(inputs=base.inputs, outputs=out)
  return model

model = model()
model.summary()
sample_input = tf.ones(shape=(BATCH_SIZE, SPECTROGRAM_5_SECOND_LENGTH, FREQUENCY_LENGTH, 2))
with tf.device("/CPU:0"):
  sample_output = model(sample_input, training=False)
print(sample_output.shape)




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
    e_all_loss.append(step)
    id += 1
  time_val.append(id)

e_val_loss = [k[0] for k in epochs_val_loss]

e_all_val_loss = []

id = 0
time_val = []
for epoch in epochs_val_loss:
  for step in epoch:
    e_all_val_loss.append(step)
    id += 1
  time_val.append(id)

plt.plot(np.arange(0, len(e_all_loss), 1), e_all_loss, label = "train loss")
# plt.plot(time_val, epochs_val_loss, label = "val loss")
plt.plot(np.arange(0, len(e_all_val_loss), 1), e_all_val_loss, label = "val loss")

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

test_id = 35
test_time_ptr = 0
time_end_point = MIN_TIME_END_POINT + test_time_ptr
df_id = int(test_time_ptr * 2)
row = test_valence_df.loc[test_id]
song_id = row["song_id"]
# Get the labels series
valence_labels = test_valence_df.loc[test_id, test_valence_df.columns != "song_id"]
arousal_labels = test_arousal_df.loc[test_id, test_arousal_df.columns != "song_id"]

valence_val = tanh_to_sigmoid(valence_labels.iloc[[df_id]].to_numpy())
arousal_val = tanh_to_sigmoid(arousal_labels.iloc[[df_id]].to_numpy())

label = tf.convert_to_tensor([valence_val, arousal_val], dtype=tf.float32)
print(f"Label: Valence: {valence_val}, Arousal: {arousal_val}")
song_path = os.path.join(AUDIO_FOLDER, str(int(song_id)) + SOUND_EXTENSION)
audio_file = tf.io.read_file(song_path)
waveforms, _ = tf.audio.decode_wav(contents=audio_file)
waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)


end_wave_index = int(time_end_point * DEFAULT_FREQ)
start_wave_index = int(end_wave_index - WINDOW_SIZE)

current_waveforms = waveforms[start_wave_index: end_wave_index, ...]

spectrograms = None
# Loop through each channel
for i in range(current_waveforms.shape[-1]):
  # Shape (timestep, frequency, 1)
  spectrogram = get_spectrogram(current_waveforms[..., i], input_len=current_waveforms.shape[0])
  if spectrograms == None:
    spectrograms = spectrogram
  else:
    spectrograms = tf.concat([spectrograms, spectrogram], axis=-1)

spectrograms = spectrograms[tf.newaxis, ...]


## Eval
y_pred_list = debugging_model(spectrograms, training=False)
print(f"Predicted y_pred value: Valence: {y_pred_list[-1][0, 0]}, Arousal: {y_pred_list[-1][0, 1]}")

plot_and_play(waveforms, time_end_point - WINDOW_TIME, WINDOW_TIME, 0)


# def show_color_mesh(spectrogram):
#   """ Generate color mesh

#   Args:
#       spectrogram (2D array): Expect shape (Frequency length, time step)
#   """
#   assert len(spectrogram.shape) == 2
#   log_spec = np.log(spectrogram + np.finfo(float).eps)
#   height = log_spec.shape[0]
#   width = log_spec.shape[1]
#   X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
#   Y = range(height)
#   plt.pcolormesh(X, Y, log_spec)
#   plt.show()

# show_color_mesh(tf.transpose(spectrograms[0, :, :, 0], [1,0]))


# %%

f, axarr = plt.subplots(7,4, figsize=(25,15))
CONVOLUTION_NUMBER_LIST = [2, 3, 4, 5]
LAYER_LIST = [3, 5, 7, 9, 10, 11, 13]
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
  # plot_spectrogram(tf.transpose(f7[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[6,x])
  plot_spectrogram(f7[0, : , :].numpy(), axarr[6,x])
  axarr[6,x].grid(False)
  # f8 = y_pred_list[LAYER_LIST[7]]
  # plot_spectrogram(tf.transpose(f8[0, : , :, CONVOLUTION_NUMBER], [1,0]).numpy(), axarr[7,x])
  # axarr[7,x].grid(False)

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

