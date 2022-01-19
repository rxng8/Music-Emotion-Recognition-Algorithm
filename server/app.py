import uvicorn
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("./model/my_model/")

DEFAULT_FREQ = 44100
DEFAULT_TIME = 45
WAVE_ARRAY_LENGTH = DEFAULT_FREQ * DEFAULT_TIME
FREQUENCY_LENGTH = 129
N_CHANNEL = 2
SPECTROGRAM_TIME_LENGTH = 15502

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

def predict(sound: bytes):
  waveforms, _ = tf.audio.decode_wav(contents=sound)
  # Pad to max 45 second. Shape (total_frequency, n_channels)
  waveforms = preprocess_waveforms(waveforms, WAVE_ARRAY_LENGTH)
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


  padded_spectrogram = np.zeros((SPECTROGRAM_TIME_LENGTH, FREQUENCY_LENGTH, N_CHANNEL), dtype=float)
  # spectrograms = spectrograms[tf.newaxis, ...]
  # some spectrogram are not the same shape
  padded_spectrogram[:spectrograms.shape[0], :spectrograms.shape[1], :] = spectrograms
  
  sample_input = tf.convert_to_tensor(padded_spectrogram)
  prediction = model(sample_input[tf.newaxis, ...], training=False)[0, ...]
  return prediction

@app.post("/predict/sound")
async def predict_api(file: UploadFile = File(...)):
  extension = file.filename.split(".")[-1] in ("wav")
  if not extension:
    return "Sound must be wav format!"
  sound = await file.read()
  prediction = predict(sound)
  # print(prediction)
  return str(prediction)

if __name__ == "__main__":
  uvicorn.run(app, debug=True)