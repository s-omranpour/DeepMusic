# DeepNote
DeepNote is a high level python package with following features:
 - supporting different formats like MIDI, REMI, Compound Word.
 - representing musical data in a very simple but useful way for high level music theoretic manipulations.
 - preprocessing musical data in order to feed them to neural networks (chord extraction, quantization and numericalization).
 - correcting notes' pitches based on given chords (generated music post-processing).

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
