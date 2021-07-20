# DeepNote
DeepNote is a high level python package with following features:
 - supporting different formats like MIDI, REMI, Compound Word and pianoroll. [1, 2]
 - representing musical data in a very simple but useful way for high level music theoretic manipulations.
 - preprocessing musical data in order to feed them to neural networks (chord extraction, quantization and numericalization).
 - supporting metrics used for evaluating generated sequences. [3, 4]

## Install
`pip install deepnote`


## Usage
```python
from deepnote import MusicRepr

## reading a midi file
seq = MusicRepr.from_file('test.mid')

## displaying first 10 events
print(seq[:10])

## export to remi representation
remi = seq.to_remi(ret='token')

## export to compound word representation
cp = seq.to_cp()
print(cp.shape) ## (num_events, 8)

## splitting song's bars
bars = seq.get_bars()
print(len(bars))
```


for more details please see `examples`.

## References
[1] Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions, Yu-Siang Huang, Yi-Hsuan Yang

[2] Compound Word Transformer: Learning to Compose Full-Song Musicover Dynamic Directed Hypergraphs, Wen-Yi Hsiao, Jen-Yu Liu, Yin-Cheng Yeh, Yi-Hsuan Yang

[3] The Jazz Transformer on the Front Line: Exploring the Shortcomings of AI-composed Music through Quantitative Measures, Shih-Lun Wu, Yi-Hsuan Yang

[4] [https://github.com/slSeanWU/MusDr](https://github.com/slSeanWU/MusDr)
