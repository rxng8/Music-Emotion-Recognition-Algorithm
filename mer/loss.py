import tensorflow as tf

def simple_mse_loss(true, pred):

  # loss_valence = tf.reduce_sum(tf.square(true[..., 0] - pred[0][..., 0])) / true.shape[0] # divide by batch size
  # loss_arousal = tf.reduce_sum(tf.square(true[..., 1] - pred[1][..., 0])) / true.shape[0] # divide by batch size

  # return loss_valence + loss_arousal

  loss = tf.reduce_sum(tf.square(true - pred)) / true.shape[0]

  return loss

def simple_mae_loss(true, pred):
  loss = tf.reduce_sum(tf.abs(true - pred)) / true.shape[0]
  return loss