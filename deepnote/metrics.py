import numpy as np
from scipy.stats import entropy
from .repr import MusicRepr

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
        piece = MusicRepr.concatenate(bars[idx:idx+window]).to_pianoroll(separate_tracks=False, binarize=True)
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


        
    