import tensorflow as tf

def simple_mse_loss(true, pred):
  return tf.nn.l2_loss(true - pred) / true.shape[0] # divide by batch size