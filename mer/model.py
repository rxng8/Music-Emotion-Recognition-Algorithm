import tensorflow as tf
import tensorflow.keras.layers as L

from .const import *

class SimpleDenseModel(tf.keras.Model):
  def __init__(self, max_timestep, n_freq, n_channel, batch_size, **kwargs):
    super().__init__(**kwargs)
    self.max_timestep = max_timestep
    self.n_freq = n_freq
    self.n_channel = n_channel
    self.batch_size = batch_size

    self.resize = tf.keras.layers.Resizing(self.n_freq, 1024)
    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(512, activation="relu")
    self.dense2 = tf.keras.layers.Dense(256, activation="relu")
    self.dense3 = tf.keras.layers.Dense(128, activation="relu")
    self.dense4 = tf.keras.layers.Dense(64, activation="relu")
    self.dense5 = tf.keras.layers.Dense(2, activation="relu")

  def call(self, x):
    """ 

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Condense
    tensor = self.resize(x)
    tensor = self.flatten(tensor)
    tensor = self.dense1(tensor)
    tensor = self.dense2(tensor)
    tensor = self.dense3(tensor)
    tensor = self.dense4(tensor)
    out = self.dense5(tensor)
    return out
  
  def model(self):
    x = tf.keras.layers.Input(shape=(self.max_timestep, self.n_freq, self.n_channel))
    return tf.keras.Model(inputs=x, outputs=self.call(x))

class ConvBlock(tf.keras.Model):
  def __init__(self, neurons, **kwargs) -> None:
    super().__init__(**kwargs)
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(neurons, (3,3), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(neurons // 2, (1,1), padding="same"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2,2),
      tf.keras.layers.Dropout(0.1)
    ])

  def call(self, x):
    return self.model(x)
  

class SimpleConvModel(tf.keras.Model):
  def __init__(self, max_timestep, n_freq, n_channel, batch_size, **kwargs):
    super().__init__(**kwargs)
    self.max_timestep = max_timestep
    self.n_freq = n_freq
    self.n_channel = n_channel
    self.batch_size = batch_size

    neuron_conv = [64, 128, 256, 512, 1024]

    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Resizing(self.n_freq, 512, input_shape=(max_timestep, n_freq, n_channel)))
    for neuron in neuron_conv:
      self.model.add(ConvBlock(neuron))
    self.model.add(tf.keras.layers.Flatten())
    self.model.add(tf.keras.layers.Dense(128, activation="relu"))
    self.model.add(tf.keras.layers.Dense(2, activation="relu"))
    self.model.summary() 

  def call(self, x):
    """ 

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Condense
    return self.model(x)
  
  def model(self):
    x = tf.keras.layers.Input(shape=(self.max_timestep, self.n_freq, self.n_channel))
    return tf.keras.Model(inputs=x, outputs=self.call(x))


class ConvBlock2(tf.keras.Model):
  def __init__(self, neurons, **kwargs) -> None:
    super().__init__(**kwargs)
    self.model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(neurons, (5,5), padding="valid"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.Conv2D(neurons // 2, (1,1), padding="valid"),
      tf.keras.layers.LeakyReLU(alpha=0.1),
      tf.keras.layers.MaxPool2D(2,2),
      tf.keras.layers.Dropout(0.1)
    ])

  def call(self, x):
    return self.model(x)

def Simple_CRNN():
  """[summary]

  Args:
    inputs (tf.Tensor): Expect tensor shape (batch, width, height, channel)

  Returns:
    [type]: [description]
  """
  inputs = L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = L.Resizing(FREQUENCY_LENGTH, 1024)(tensor)
  
  tensor = L.Conv2D(64, (5,5), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(64 // 2, (1,1), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(128, (5,5), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(128 // 2, (1,1), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(256, (5,5), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(256 // 2, (1,1), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(512, (5,5), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(512 // 2, (1,1), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.MaxPool2D(pool_size=(2,1), strides=(2,1))(tensor)
  tensor = L.Conv2D(512, (2,2), padding="valid")(tensor)
  tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Reshape((59, 512))(tensor)
  tensor = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  tensor = L.Bidirectional(L.LSTM(64, return_sequences=True))(tensor)
  tensor = L.Bidirectional(L.LSTM(32))(tensor)
  tensor = L.Dense(128, activation="relu")(tensor)
  out = L.Dense(2, activation="relu")(tensor)

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model

def Simple_CRNN_2():
  """[summary]

  Args:
    inputs (tf.Tensor): Expect tensor shape (batch, width, height, channel)

  Returns:
    [type]: [description]
  """
  inputs = L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = L.Resizing(FREQUENCY_LENGTH, 1024)(tensor)
  
  tensor = L.Conv2D(64, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(64 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(128, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(128 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(256, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(256 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(512, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(512 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Permute((2, 1, 3))(tensor)
  tensor = L.Reshape((60, 4 * 256))(tensor)

  # tensor = L.MaxPool2D(pool_size=(2,1), strides=(2,1))(tensor)
  # tensor = L.Conv2D(512, (2,2), padding="valid")(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  # out = L.Dropout(0.1)(tensor)

  # tensor = L.Bidirectional(L.LSTM(128, return_sequences=True, activation="sigmoid"))(tensor)
  # tensor = L.Bidirectional(L.LSTM(128, return_sequences=True, activation="sigmoid"))(tensor)
  tensor = L.Bidirectional(L.LSTM(128, activation="tanh"))(tensor)
  tensor = L.Dense(512, activation="relu")(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  out = L.Dense(2, activation="relu")(tensor)

  # tensor_1 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  # tensor_1 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor_1)
  # tensor_1 = L.Bidirectional(L.LSTM(128))(tensor_1)
  # tensor_1 = L.Dense(512, activation="relu")(tensor_1)
  # tensor_1 = L.Dense(256, activation="relu")(tensor_1)
  # tensor_1 = L.Dense(64, activation="relu")(tensor_1)
  # out_1 = L.Dense(1, activation="relu")(tensor_1)

  # tensor_2 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  # tensor_2 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor_2)
  # tensor_2 = L.Bidirectional(L.LSTM(128))(tensor_2)
  # tensor_2 = L.Dense(512, activation="relu")(tensor_2)
  # tensor_2 = L.Dense(256, activation="relu")(tensor_2)
  # tensor_2 = L.Dense(64, activation="relu")(tensor_2)
  # out_2 = L.Dense(1, activation="relu")(tensor_2)
  

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model


def Simple_CRNN_3():
  """ CRNN that uses GRU

  Args:
    inputs (tf.Tensor): Expect tensor shape (batch, width, height, channel)

  Returns:
    [type]: [description]
  """
  inputs = L.Input(shape=(SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, 2))
  tensor = L.Permute((2, 1, 3))(inputs)
  tensor = L.Resizing(FREQUENCY_LENGTH, 1024)(tensor)
  
  tensor = L.Conv2D(64, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(64 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(128, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(128 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(256, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(256 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Conv2D(512, (5,5), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.Conv2D(512 // 2, (1,1), padding="valid")(tensor)
  tensor = L.ReLU()(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  tensor = L.MaxPool2D(2,2)(tensor)
  tensor = L.Dropout(0.1)(tensor)

  tensor = L.Permute((2, 1, 3))(tensor)
  tensor = L.Reshape((60, 4 * 256))(tensor)

  # tensor = L.MaxPool2D(pool_size=(2,1), strides=(2,1))(tensor)
  # tensor = L.Conv2D(512, (2,2), padding="valid")(tensor)
  # tensor = L.LeakyReLU(alpha=0.1)(tensor)
  # out = L.Dropout(0.1)(tensor)

  tensor = L.GRU(256, activation="tanh", return_sequences=True)(tensor)
  tensor = L.GRU(128, activation="tanh", return_sequences=True)(tensor)
  tensor = L.GRU(64, activation="tanh")(tensor)
  tensor = L.Dense(512, activation="relu")(tensor)
  tensor = L.Dense(256, activation="relu")(tensor)
  tensor = L.Dense(64, activation="relu")(tensor)
  out = L.Dense(2, activation="relu")(tensor)

  # tensor_1 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  # tensor_1 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor_1)
  # tensor_1 = L.Bidirectional(L.LSTM(128))(tensor_1)
  # tensor_1 = L.Dense(512, activation="relu")(tensor_1)
  # tensor_1 = L.Dense(256, activation="relu")(tensor_1)
  # tensor_1 = L.Dense(64, activation="relu")(tensor_1)
  # out_1 = L.Dense(1, activation="relu")(tensor_1)

  # tensor_2 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor)
  # tensor_2 = L.Bidirectional(L.LSTM(128, return_sequences=True))(tensor_2)
  # tensor_2 = L.Bidirectional(L.LSTM(128))(tensor_2)
  # tensor_2 = L.Dense(512, activation="relu")(tensor_2)
  # tensor_2 = L.Dense(256, activation="relu")(tensor_2)
  # tensor_2 = L.Dense(64, activation="relu")(tensor_2)
  # out_2 = L.Dense(1, activation="relu")(tensor_2)
  

  model = tf.keras.Model(inputs=inputs, outputs=out)
  return model