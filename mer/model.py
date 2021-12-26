import tensorflow as tf

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