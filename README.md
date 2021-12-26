# MSA

* convert to wav code: 

```
cd audio
mkdir wav
for %f in (*.mp3) do (ffmpeg -i "%f" -acodec pcm_s16le  -ar 44100  "./wav/%f.wav")
```
or
```
python wav_converter.py {src} {dst} {ffmpeg bin path}
# E.g: python wav_converter.py "./dataset/DEAM/audio" "./dataset/DEAM/wav" "C:/Users/Alex Nguyen/Documents/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
```

## Resources

* Database benchmark: 

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173392

https://www.kaggle.com/imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music
https://cvml.unige.ch/databases/DEAM/
https://cvml.unige.ch/databases/DEAM/manual.pdf

https://www.semanticscholar.org/paper/The-AMG1608-dataset-for-music-emotion-recognition-Chen-Yang/16a21a57c85bae0ada26454300dc5c5891f1c0e2

The PMEmo Dataset for Music Emotion Recognition. https://dl.acm.org/doi/10.1145/3206025.3206037
https://github.com/HuiZhangDB/PMEmo

RAVDESS database: https://zenodo.org/record/1188976

* Technical: 

https://www.tensorflow.org/io/tutorials/audio
https://librosa.org/
https://www.tensorflow.org/tutorials/audio/simple_audio

* Related work and works

most influlece library mir search: https://www.semanticscholar.org/search?fos%5B0%5D=computer-science&q=Music%20Emotion%20Recognition&sort=influence

https://www.frontiersin.org/articles/10.3389/fpsyg.2021.760060/full

label scheme: Lin, Y. C., Yang, Y. H., and Homer, H. (2011). Exploiting online music tags for music emotion classification. ACM Trans. Multimed. Comput. Commun. Appl. 7, 1–16. doi: 10.1145/2037676.2037683

novel features of music for mer: https://www.semanticscholar.org/paper/Novel-Audio-Features-for-Music-Emotion-Recognition-Panda-Malheiro/6feb6c070313992897140a1802fdb8f0bf129422


musical texture and espresitivity features: https://www.semanticscholar.org/paper/Musical-Texture-and-Expressivity-Features-for-Music-Panda-Malheiro/e4693023ae525b7dd1ecabf494654e7632f148b3

specch recognition: http://proceedings.mlr.press/v32/graves14.pdf

data augmentation: https://paperswithcode.com/paper/specaugment-a-simple-data-augmentation-method

rnn regularization technique: https://paperswithcode.com/paper/recurrent-neural-network-regularization

wave to vec: https://paperswithcode.com/paper/wav2vec-2-0-a-framework-for-self-supervised

* MER Task:

Music Mood Detection Based On Audio And Lyrics With Deep Neural Net: https://paperswithcode.com/paper/music-mood-detection-based-on-audio-and
Transformer-based: https://paperswithcode.com/paper/transformer-based-approach-towards-music

tutorial ismir: https://arxiv.org/abs/1709.04396

crnn based?