import os
from os import path
import sys
from pydub import AudioSegment

if __name__ == "__main__":

  src = sys.argv[1]
  dst = sys.argv[2]
  AudioSegment.converter = sys.argv[3]

  for i, fname in enumerate(os.listdir(src)):
    try:
      # convert mp3 to wav
      sound = AudioSegment.from_mp3(os.path.join(src, fname))
      sound.export(os.path.join(dst, fname[:-4] + ".wav"), format="wav")
      print(f"Exported to {os.path.join(dst, fname[:-4] + '.wav')}" )
    except:
      print(f"Cannot convert {fname} to wav.")
