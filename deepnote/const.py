import numpy as np
import itertools

## general
DEFAULT_UNIT = 12
DEFAULT_TICK_RESOL = 120
DEFAULT_STEP = DEFAULT_TICK_RESOL // DEFAULT_UNIT
DEFAULT_BAR_RESOL = DEFAULT_TICK_RESOL * 4
DEFAULT_INSTRUMENTS = [
    'piano',
    'percussion',
    'organ',
    'guitar',
    'bass',
    'strings',
    'ensemble',
    'brass',
    'reed',
    'pipe',
    'synth-lead',
    'synth-pad',
    'synth-effects',
    'ethnic',
    'percussive',
    'sound-effects',
    'drums'
]

## note
DEFAULT_POS_BINS = np.arange(1, 4*DEFAULT_UNIT+1)
DEFAULT_DUR_BINS = np.arange(1, 4*DEFAULT_UNIT+1)
DEFAULT_VEL_BINS = np.linspace(9, 127, 30, dtype=int)

## metric
DEFAULT_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEFAULT_CHORD_QUALITIES = ['M', 'm', 'o', '+', '7', 'M7', 'm7', 'o7', '/o7', 'sus2', 'sus4']
DEFAULT_CHORDS = ['_'.join(e) for e in itertools.product(DEFAULT_PITCH_CLASSES, DEFAULT_CHORD_QUALITIES)]
DEFAULT_TEMPO_BINS = np.linspace(42, 296, 20, dtype=int)

## remi
DEFAULT_TOKENS = ['Bar', 'EOS'] +\
    ['BeatPosition_'+str(p) for p in DEFAULT_POS_BINS]+ \
        ['BeatTempo_'+str(t) for t in DEFAULT_TEMPO_BINS]+ \
            ['BeatChord_'+str(c) for c in DEFAULT_CHORDS] + \
                ['NoteInstFamily_'+str(inst) for inst in DEFAULT_INSTRUMENTS] + \
                    ['NotePitch_'+str(i) for i in range(128)] + \
                        ['NoteDuration_'+str(d) for d in DEFAULT_DUR_BINS] + \
                            ['NoteVelocity_'+str(v)for v in DEFAULT_VEL_BINS]