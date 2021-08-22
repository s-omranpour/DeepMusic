from deepnote.modules import Metric, Note
import numpy as np
from scipy.stats import entropy
import itertools

from .repr import MusicRepr
from .scale import Scale

def pitch_histogram_entropy(seq : MusicRepr, window : int = 1, pitch_class: bool = False, return_probs=True):
    """
    seq : input sequence
    window : number of bars as window
    """
    bars = seq.get_bars()
    n = len(bars)
    if n < window:
        window = n
        print(f'[Warning] Window size set to {window}.')
    
    ents = []
    probs = []
    for idx in range(0, n-window+1):
        piece = MusicRepr.concatenate(bars[idx:idx+window]).to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False)
        pitch_acts = piece.sum(axis=1)
        if pitch_class:
            res = []
            for i in range(12):
                res += [pitch_acts[i::12].sum()]
            pitch_acts = np.array(res)
        prob = pitch_acts / pitch_acts.sum() if pitch_acts.sum() > 0 else pitch_acts
        ents += [entropy(prob) / np.log(2)]
        probs += [prob]
    if return_probs:
        return np.array(ents), np.array(probs).T
    return np.array(ents)


def polyphony(seq : MusicRepr):
    pitch_ons = seq.to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False).sum(axis=0)
    return pitch_ons.sum()/np.sum(pitch_ons > 0)

def polyphony_rate(seq : MusicRepr):
    pitch_ons = seq.to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False).sum(axis=0)
    return np.sum(pitch_ons > 0) / pitch_ons.shape[0]

def pitch_in_scale_rate(seq : MusicRepr):
    scale = Scale()
    pianoroll = seq.to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False)

    scores = {}
    prev_idx = 0
    prev_chord = None
    for idx, e in enumerate(seq.events):
        if isinstance(e, Metric) and e.chord is not None:
            if prev_chord is None:
                prev_chord = e.chord
            elif e.chord != prev_chord:
                if prev_chord not in scores:
                    scores[prev_chord] = {'score': 0, 'n_pitches' : 0}
                scores[prev_chord]['score'] += scale.pitch_in_scale(pianoroll[:, prev_idx:idx], chord=prev_chord)
                scores[prev_chord]['n_pitches'] += np.sum(pianoroll[:, prev_idx:idx] > 0)
                prev_chord = e.chord
                prev_idx = idx
    for chord in scores:
        scores[chord] = scores[chord]['score'] / scores[chord]['n_pitches'] if scores[chord]['n_pitches'] > 0 else 0.
    return scores

def empty_beat_rate(seq: MusicRepr):
    note_ons = seq.to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False).sum(axis=0)
    return np.sum(note_ons == 0) / note_ons.shape[0]

def grooving_pattern_similarity(seq : MusicRepr):
    def xor_distance(bar1, bar2):
        onsets1 = bar1.to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False).sum(axis=0)
        onsets2 = bar2.to_pianoroll(separate_tracks=False, binarize=True, add_tempo_chord=False).sum(axis=0)
        return 1 - np.logical_xor(onsets1 > 0, onsets2 > 0).sum() / onsets1.shape[0]

    bars = seq.get_bars()
    scores = []
    for pair in itertools.combinations(range(len(bars)), 2):
        idx1, idx2 = pair
        scores += [xor_distance(bars[idx1], bars[idx2])]
    return np.mean(scores)

def chord_progression_irregularity(seq : MusicRepr, ngram=3, ret_unique_ngrams=False):
    chords = [e.chord for e in list(filter(lambda x: isinstance(x, Metric), seq.events)) if e.chord is not None]
    num_ngrams = len(chords) - ngram
    unique_set = set()
    for i in range(num_ngrams):
        word = '|'.join(chords[i:i+ngram])
        unique_set.update([word])
    res = len(unique_set) / num_ngrams
    if ret_unique_ngrams:
        return res, unique_set
    return res


def structuredness_indicator(seq : MusicRepr): 
    raise NotImplementedError