from typing import List, Dict, Union
from collections import Counter
import numpy as np
import os
import logging
from copy import deepcopy

from miditoolkit.midi.parser import MidiFile
from miditoolkit.midi import containers as ct
from midi2audio import FluidSynth
from deepmusic import conf

from deepmusic.conf import INSTRUMENT_FAMILIES, CHORDS, MusicConfig
from deepmusic.event import NoteEvent, TempoEvent, ChordEvent
from deepmusic import utils


class Music:
    def __init__(self, tracks : List = [], tempos : List[TempoEvent] = [], chords : List[ChordEvent] = [], config : MusicConfig = None, name : str = ''):
        self.config = config
        self.name = name
        self.tracks = []
        self.tempos = []
        self.chords = []
        
        if len(tracks):
            configs = list(set([t.config for t in tracks]))
            assert len(configs) == 1, "Inconsistancy among inserted tracks' configs."
            if self.config is not None:
                assert self.config == configs[0], "Inconsistancy among tracks' configs and entered config."
            self.config = configs[0]
            self.tracks = tracks

        if len(tempos) + len(chords):
            assert self.config is not None, "Please enter a valid config when no tracks are entered."
            self.tempos = tempos
            self.chords = chords

        if self.config is None:
            self.config = MusicConfig()
        self.clean()
        

    @staticmethod
    def from_file(
        file_path : str, 
        config: MusicConfig = None, 
        unit: int = 12,
        min_tempo : int = 30,
        max_tempo : int = 300,
        num_tempo_bins : int = 30, 
        num_velocity_bins : int = 30):

        midi = MidiFile(file_path)
        midi = deepcopy(midi)
        tick_resol = midi.ticks_per_beat
        
        if config is None:
            config = MusicConfig(unit, tick_resol, min_tempo, max_tempo, num_tempo_bins, num_velocity_bins)
        else:
            config.update_resolution(tick_resol)

        tracks = [Track.create_track_from_midi_instrument(config, inst) for inst in midi.instruments]
        tempos = []
        for tempo in midi.tempo_changes:
            time = tempo.time // config.step
            tempo = np.argmin(np.abs(config.tempo_bins - int(tempo.tempo)))
            tempos += [TempoEvent(bar=time // config.n_bar_steps, beat=time % config.n_bar_steps, tempo=tempo)]

        chords = []
        for marker in midi.markers:
            if marker.text.startswith('Chord_'):
                time = marker.time // config.step
                chord = CHORDS.index(marker.text[6:]) 
                chords += [ChordEvent(bar=time // config.n_bar_steps, beat=time % config.n_bar_steps, chord=chord)]

        return Music(tracks, tempos, chords, config, file_path)

    @staticmethod
    def from_pianoroll(rolls : Dict[int, np.array], config : MusicConfig):
        if config is None:
            config = MusicConfig()
        tracks = [Track.from_pianoroll(v, k, config) for k,v in rolls.items()]
        return Music(tracks, config=config)

    @staticmethod
    def from_indices(indices : List, config : MusicConfig = None):
        if config is None:
            config = MusicConfig()
        return Music.from_tokens(config.decode(indices), config)

    @staticmethod
    def from_tokens(tokens : List, config : MusicConfig = None):
        if config is None:
            config = MusicConfig()

        tracks = {}
        tempos = []
        chords = []
        prev_pos_tok = None
        nb = -1
        for idx,tok in enumerate(tokens):
            if tok.startswith('Bar'):
                nb += 1
            elif tok.startswith('Beat'):
                prev_pos_tok = tok
            elif tok.startswith('Tempo'):
                tempos += [TempoEvent.from_tokens([prev_pos_tok, tok], bar=nb)]
            elif tok.startswith('Chord'):
                chords += [ChordEvent.from_tokens([prev_pos_tok, tok], bar=nb)]
            elif tok.startswith('NoteInstrument'):
                program = int(tok[15:])
                if program not in tracks:
                    tracks[program] = []
                tracks[program] += [NoteEvent.from_tokens([prev_pos_tok] + tokens[idx+1:idx+4], bar=nb)] 
        tracks = [Track(k, v, config) for k,v in tracks.items()]
        return Music(tracks, tempos, chords, config)

    @staticmethod
    def from_tuples(tuples : Union[List[List], np.array], config : MusicConfig = None):
        tuples = np.array(tuples)
        assert len(tuples.shape) == 2 and tuples.shape[1] == 7, "Invalid tuples shape."
        if config is None:
            config = MusicConfig()
        
        tracks = []
        programs = set(tuples[tuples[:, -1] > 0][:, -1]) ## program = 0 means not a note
        for program in programs:
            track = tuples[tuples[:, -1] == program][:, :-1]
            notes = [NoteEvent.from_tuple(t) for t in track]
            tracks += [Track(program, notes, config)]

        tempos = []
        for time in np.where(tuples[:,2] > 0)[0]:
            bar, beat, tempo = tuples[time, :3]
            tempos += [TempoEvent(bar, beat, tempo-1)]

        chords = []
        for time in np.where(tuples[:,3] > 0)[0]:
            bar, beat, chord = tuples[time, :3]
            chords += [ChordEvent(bar, beat, chord-1)]
        return Music(tracks, tempos, chords, config)

    @staticmethod
    def merge(musics : List, merge_similar_programs=True):
        """
            if merge == True:
                merge tracks with same program and different names
            else:
                merge tracks with same program and name
        """
        ## check for config consistency
        configs = list(set([m.config for m in musics]))
        assert len(configs) == 1, "Inconsistancy among inserted musics' configs."
        config = configs[0]

        tracks = utils.flatten([m.get_tracks() for m in musics])
        tempos = utils.sort_and_remove_identical_events(utils.flatten([m.get_tempos() for m in musics]))
        chords = utils.sort_and_remove_identical_events(utils.flatten([m.get_chords() for m in musics]))
        res = Music(tracks, tempos, chords, config, name='Merged : '+', '.join([m.name for m in musics]))
        if merge_similar_programs:
            ids = set([(t.program, t.name) for t in res.tracks])
            for id in ids:
                indices = [idx for idx in range(len(res.tracks)) if res.tracks[idx].program == id[0] and res.tracks[idx].name == id[1]]
                res.merge_tracks(indices, name=id[1])
        return res

    @staticmethod
    def concatenate(musics : List, merge_similar_tracks=True):
        """
            if merge == True:
                merge tracks with same program and different names
            else:
                merge tracks with same program and name
        """
        assert len(musics) > 0, "Not enough musics to concatenate."
        configs = list(set([m.config for m in musics]))
        assert len(configs) == 1, "Inconsistancy among inserted musics' configs."
        config = configs[0]
        all_tracks = []
        tempos = []
        chords = []
        n_prev_bars = 0
        for music in musics:
            m = deepcopy(music)
            m.pad_left(n_prev_bars)
            n_prev_bars = m.get_bar_count()
            all_tracks += m.get_tracks()
            tempos += m.get_tempos()
            chords += m.get_chords()
            
            
        tempos = utils.sort_and_remove_identical_events(tempos)
        chords = utils.sort_and_remove_identical_events(chords)
        res = Music(all_tracks, tempos, chords, config, name='Concatented : '+', '.join([m.name for m in musics]))
        if merge_similar_tracks:
            ids = set([(t.program, t.name) for t in res.tracks])
            for id in ids:
                indices = [idx for idx in range(len(res.tracks)) if res.tracks[idx].program == id[0] and res.tracks[idx].name == id[1]]
                res.merge_tracks(indices, name=id[1])
        return res

    def change_config(self, config : MusicConfig):
        for t in self.tracks:
            t.change_config(config)
        for tempo in self.tempos:
            tempo.tempo = np.argmin(np.abs(config.tempo_bins - self.config.tempo_bins[tempo.tempo]))
            tempo.update_metric_attributes_with_config(config, self.config)
        for chord in self.chords:
            chord.update_metric_attributes_with_config(config, self.config)
        self.config = config

    def clean(self):
        """
        1. remove duplicate and invalid tempos and chords and sort them
        2. clean each track
        3. remove empty tracks
        """
        prev_tempo = -1
        res = []
        for tempo in self.tempos:
            if utils.validate_tempo(self.config, tempo) and tempo.tempo != prev_tempo:
                prev_tempo = tempo.tempo
                res += [tempo]
        self.tempos = utils.sort_events(res)

        prev_chord = -1
        res = []
        for chord in self.chords:
            if utils.validate_chord(self.config, chord) and chord.chord != prev_chord:
                prev_chord = chord.chord
                res += [chord]
        self.chords = utils.sort_events(res)

        for t in self.tracks:
            t.clean()
        
        res = []
        for t in self.tracks:
            if not t.is_empty():
                res += [t]
        self.tracks = res

    def merge_tracks(self, indices : List, name : str = None):
        super_track = Track.merge_tracks([self.tracks[idx] for idx in indices], name=name)
        self.remove_tracks(indices=indices, inplace=True)
        self.tracks += [super_track]

    def pad_left(self, n : int):
        for track in self.tracks:
            track.pad_left(n)
        for tempo in self.tempos:
            tempo.set_metric_attributes(bar=tempo.bar + n)
        for chord in self.chords:
            chord.set_metric_attributes(bar=chord.bar + n)

    def remove_starting_silent_bars(self, inplace : bool = False):
        min_bars = [t.notes[0].bar for t in self.tracks]
        if len(self.tempos):
            min_bars += [self.tempos[0].bar]
        if len(self.chords):
            min_bars += [self.chords[0].bar]
        offset = min(min_bars) if len(min_bars) else 0
        tracks = self.get_tracks()
        for t in tracks:
            for note in t.notes:
                note.set_metric_attributes(bar=note.bar - offset)
        tempos = self.get_tempos()
        for t in tempos:
            t.set_metric_attributes(bar=t.bar - offset)
        chords = self.get_chords()
        for c in chords:
            c.set_metric_attributes(bar=c.bar - offset)
        if inplace:
            self.tracks = tracks
            self.tempos = tempos
            self.chords = chords
            return 
        return Music(tracks, tempos, chords, self.get_config(), self.name)

    def remove_tracks(self, indices:List=None, programs:List=None, inst_families:List=None, inplace : bool = False):
        tracks = self.get_tracks()
        selected_tracks = []
        if indices is not None:
            selected_tracks = [tracks[idx] for idx in indices]
        if programs is not None:
            selected_tracks = [t for t in tracks if t.program in programs]
        if inst_families is not None:
            selected_tracks = [t for t in tracks if t.inst_family in inst_families]
        for t in set(selected_tracks):
            tracks.remove(t)
        if inplace:
            self.tracks = tracks
            return
        return Music(tracks, self.get_tempos(), self.get_chords(), self.get_config(), self.name)

    def keep_tracks(self, indices:List=None, programs:List=None, inst_families:List=None, inplace : bool = False):
        tracks = set()
        if indices is not None:
            for idx in indices:
                tracks.update([self.tracks[idx]]) 
        if programs is not None:
            for t in self.tracks:
                if t.program in programs:
                    tracks.update([t])
        if inst_families is not None:
            for t in self.tracks:
                if t.inst_family in inst_families:
                    tracks.update([t])
        if inplace:
            self.tracks = tracks
            return
        return Music(list(tracks), self.get_tempos(), self.get_chords(), self.get_config(), self.name)

    def find_position_index(self, events : List, bar : int, beat : int):
        for i, n in enumerate(events):
            if n.bar >= bar and n.beat >= beat:
                return i
        return -1

    def slice_by_beat(self, start : int, end : int, inplace : bool = False):
        tracks = [t.slice_by_beat(start, end) for t in self.get_tracks()]
        tracks = list(filter(lambda x: not x.is_empty(), tracks))

        s = self.find_position_index(self.tempos, start // self.config.n_bar_steps, start % self.config.n_bar_steps)
        e = self.find_position_index(self.tempos, end // self.config.n_bar_steps, end % self.config.n_bar_steps)
        if s == -1:
            tempos = []
        else:
            tempos = self.get_tempos()[s:]
            if e > -1:
                tempos = tempos[:e-s]
        
        s = self.find_position_index(self.chords, start // self.config.n_bar_steps, start % self.config.n_bar_steps)
        e = self.find_position_index(self.chords, end // self.config.n_bar_steps, end % self.config.n_bar_steps)
        if s == -1:
            chords = []
        else:
            chords = self.get_chords()[s:]
            if e > -1:
                chords = chords[:e-s]
        if inplace:
            self.tracks = tracks
            self.tempos = tempos
            self.chords = chords
            self.remove_starting_silent_bars(inplace=True)
            return
        return Music(tracks, tempos, chords, self.get_config(), name=self.name).remove_starting_silent_bars()

    def slice_by_bar(self, start : int, end : int, inplace : bool = False):
        tracks = [t.slice_by_bar(start, end) for t in self.get_tracks()]
        tracks = list(filter(lambda x: not x.is_empty(), tracks))

        s = self.find_position_index(self.tempos, start, 0)
        e = self.find_position_index(self.tempos, end, 0)
        if s == -1:
            tempos = []
        else:
            tempos = self.get_tempos()[s:]
            if e > -1:
                tempos = tempos[:e-s]
        
        s = self.find_position_index(self.chords, start, 0)
        e = self.find_position_index(self.chords, end, 0)
        if s == -1:
            chords = []
        else:
            chords = self.get_chords()[s:]
            if e > -1:
                chords = chords[:e-s]
        if inplace:
            self.tracks = tracks
            self.tempos = tempos
            self.chords = chords
            self.remove_starting_silent_bars(inplace=True)
            return
        return Music(tracks, tempos, chords, self.config, name=self.name).remove_starting_silent_bars()

    def get_tracks(self):
        return deepcopy(self.tracks)

    def get_tempos(self):
        return deepcopy(self.tempos)

    def get_chords(self):
        return deepcopy(self.chords)

    def get_config(self):
        return deepcopy(self.config)

    def get_bars(self):
        bars = [t.get_bars() for t in self.get_tracks()]
        tempos = utils.organize_events_by_attr(self.tempos, ['bar'])
        chords = utils.organize_events_by_attr(self.chords, ['bar'])
        res = {}
        for i in range(self.get_bar_count()):
            res[i] = {'tracks' : [], 'tempos': [], 'chords' : []}
            for track_bars in bars:
                if i in track_bars:
                    res[i]['tracks'] += [track_bars[i]]
            if i in tempos:
                res[i]['tempos'] = tempos[i]
            if i in chords:
                res[i]['chords'] = chords[i]
            if len(res[i]['tracks']) + len(res[i]['tempos']) + len(res[i]['chords']):
                res[i] = Music(
                    tracks=res[i]['tracks'], 
                    tempos=res[i]['tempos'],
                    chords=res[i]['chords'],
                    config=self.config,
                    name=self.name + ' - Bar No. '+str(i)
                ).remove_starting_silent_bars()
            else:
                del res[i]
        return res

    def get_bar_count(self):
        n_bars = [t.get_bar_count() for t in self.tracks]
        if len(self.tempos):
            n_bars += [self.tempos[-1].bar + 1]
        if len(self.chords):
            n_bars += [self.chords[-1].bar + 1]
        return max(n_bars)

    def get_num_notes(self):
        return sum([len(t) for t in self.tracks])

    def get_instruments(self, family=False):
        return list(set([t.inst_family if family else t.program for t in self.tracks]))

    def to_tokens(self, add_tempo_chord=True, return_indices=False, add_bos=False, add_eos=False):
        res = []
        tracks = [t.organize() for t in self.tracks]
        tempos = utils.organize_events_by_attr(self.tempos, ['bar', 'beat'])
        chords = utils.organize_events_by_attr(self.chords, ['bar', 'beat'])
        for bar_idx in range(self.get_bar_count()):
            res += ['Bar']
            for beat in range(self.config.n_bar_steps):
                if add_tempo_chord:
                    if (bar_idx,beat) in tempos:
                        res += tempos[(bar_idx, beat)][-1].to_tokens(include_metrics=True)
                    if (bar_idx,beat) in chords:
                        res += chords[(bar_idx, beat)][-1].to_tokens(include_metrics=True)
                for track in tracks:
                    if bar_idx in track and beat in track[bar_idx]:
                        toks = track[bar_idx][beat].to_tokens(add_instrument_token=True)
                        toks = list(filter(lambda x: not x.startswith('Bar'), toks)) ## remove bar tokens
                        res += toks
        res = utils.remove_duplicate_beats_from_tokens(res)
        if add_bos:
            res = [self.config.special_tokens[0]] + res
        if add_eos:
            res += [self.config.special_tokens[1]]
        return self.config.encode(res) if return_indices else res

    def to_tuples(self, add_tempo_chord=True):
        res = np.zeros(shape=(1,8))
        tracks = [t.organize() for t in self.tracks]
        tempos = utils.organize_events_by_attr(self.tempos, ['bar', 'beat'])
        chords = utils.organize_events_by_attr(self.chords, ['bar', 'beat'])
        for bar_idx in range(self.get_bar_count()):
            for beat in range(self.config.n_bar_steps):
                has_tempo_or_chord = False
                if add_tempo_chord:
                    metric_tuple = np.array([bar_idx, beat] + [0]*6) ## last column corresponds to instrumnt. if 0 means there is no notes and only tempo or chord on this timestep
                    if (bar_idx, beat) in tempos:
                        metric_tuple[2] = tempos[(bar_idx, beat)][-1].tempo + 1 ## in tuples 0 = ignore
                    if (bar_idx, beat) in chords:
                        metric_tuple[3] = chords[(bar_idx, beat)][-1].chord + 1 ## in tuples 0 = ignore
                    if sum(metric_tuple[2:]): ## append it only if its not empty
                        res = np.concatenate([res, metric_tuple[None, :]], axis=0)
                for track in tracks:
                    if bar_idx in track and beat in track[bar_idx]:
                        tuples = track[bar_idx][beat].to_tuples(add_instrument_token=True)
                        tuples = np.concatenate([tuples[:, :2], np.zeros(shape=(tuples.shape[0], 2)), tuples[:, 2:]], axis=1) ## third and fourth columns correcspond to tempo and chord
                        res = np.concatenate([res, tuples], axis=0)
        return res[1:]

    def to_pianoroll(self, binarize=False):
        return dict(
            zip(
                [(t.program, t.name) for t in self.tracks],
                [t.to_pianoroll(binarize=binarize) for t in self.tracks]
            )
        )

    def to_midi(self, output_path : str = None):
        instruments, max_ticks = list(zip(*[t.to_midi_instrument() for t in self.tracks]))
        midi = MidiFile()
        midi.ticks_per_beat = self.config.tick_resol
        midi.instruments = instruments
        midi.max_tick = max(max_ticks)
        
        tempo_changes = []
        for e in self.tempos:
            tempo = self.config.tempo_bins[e.tempo] 
            tempo_changes += [ct.TempoChange(tempo, e.beat*self.config.step + e.bar*self.config.bar_resol)]
        chords = []
        for e in self.chords:
            chords += [ct.Marker('Chord_'+e.chord_name, e.beat*self.config.step + e.bar*self.config.bar_resol)]

        midi.tempo_changes = sorted(tempo_changes, key=lambda x: x.time)
        midi.markers = sorted(chords, key=lambda x: x.time)
        midi.key_signature_changes = []
        midi.time_signature_changes = [ct.TimeSignature(4, 4, 0)]
        if output_path:
            midi.dump(output_path)
        return midi

    def to_audio(self, audio_path, sf2_path=None):
        self.to_midi('test.mid')
        FluidSynth(sf2_path).midi_to_audio('test.mid', audio_path, sf2_path=sf2_path)
        os.remove('test.mid')

    def __repr__(self):
        str_tracks ='\n    '.join([str(t) for t in self.tracks])
        if str_tracks:
            str_tracks = '\n    ' + str_tracks + '\n  '
        str_tempos = ''
        if len(self.tempos):
            str_tempos = '\n    ' + str(self.tempos[0]) + '\n  '
            if len(self.tempos) > 1:
                str_tempos += '  ...\n  '

        str_chords = ''
        if len(self.chords):
            str_chords = '\n    ' + str(self.chords[0])
            if len(self.chords) > 1:
                str_chords += '\n    ...\n  '
        return f"Music(\n  name={self.name}, \n  bars={self.get_bar_count()},\n  tracks=[{str_tracks}],\n  tempos=[{str_tempos}],\n  chords=[{str_chords}]\n)"

    def __hash__(self):
        return hash((self.config, *self.tempos, *self.chords, *self.tracks))

    def __eq__(self, o: object):
        if isinstance(o, Music):
            return self.config == o.config and\
                set(self.tempos) == set(o.tempos) and\
                    set(self.chords) == set(o.chords) and\
                        len(self.tracks) == len(o.tracks) and\
                            all([(t in o.tracks) for t in self.tracks])

class Track:
    def __init__(self, program : int, notes: List, config : MusicConfig = None, name: str = ''):
        """
            if 0 <= program < 128:
                instrument is not a drum
            elif 128 <= program < 256:
                instrument is drum
        """
        self.notes = []
        self.config = MusicConfig() if config is None else config
        self.set_program(program)
        self.add_notes(notes)
        self.set_name(name)
        self.clean()

    @staticmethod
    def create_track_from_midi_instrument(config : MusicConfig, instrument : ct.Instrument):
        assert config is not None, "Please enter a valid config."
        assert instrument is not None and isinstance(instrument, ct.Instrument), "Invalid instrument."

        notes = []
        for note in instrument.notes:
            pos = note.start // config.step
            notes += [
                NoteEvent(
                    bar = pos // config.n_bar_steps,
                    beat = pos % config.n_bar_steps,
                    pitch=note.pitch,
                    duration=min(max((note.end - note.start) // config.step, 1), config.n_bar_steps), 
                    velocity=np.argmin(np.abs(config.velocity_bins - int(note.velocity)))
                )
            ]
        return Track(instrument.program + (128 if instrument.is_drum else 0), notes, config, instrument.name)

    @staticmethod
    def from_pianoroll(roll : np.array, program : int, config : MusicConfig = None, name : str = ''):
        config = config if config is not None else MusicConfig()
        padded = np.pad(roll[:128], ((0, 0), (1, 1)))
        diff = np.diff(padded.astype(np.int8), axis=1)
        pitches, note_ons = np.where(diff < 0)
        note_offs = np.where(diff > 0)[1]

        poses = {}
        for pitch, on, off in zip(pitches, note_ons, note_offs):
            if on not in poses:
                poses[on] = []
            poses[on] += [
                NoteEvent(
                    bar=on // config.n_bar_steps, 
                    beat=on % config.n_bar_steps, 
                    pitch=pitch, 
                    duration=off - on, 
                    velocity=int(roll[pitch, on])
                )
            ]
        return Track(program, utils.flatten([poses[t] for t in sorted(poses)], config, name))

    @staticmethod
    def merge_tracks(tracks : List, name : str = None):
        configs = list(set([t.get_config() for t in tracks]))
        assert len(configs) == 1, "Inconsistent MusicConfig among tracks."
        config = configs[0]
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        if name is None:
            name = 'Merged :' + ', '.join([t.name for t in tracks])
        return Track(program, utils.flatten([t.get_notes() for t in tracks]), config, name)
                
    @staticmethod
    def concatente(tracks: List, name : str = None):
        configs = list(set([t.get_config() for t in tracks]))
        assert len(configs) == 1, "Inconsistent MusicConfig among tracks."
        
        config = configs[0]
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        if name is None:
            name = 'Concatenated :' + ', '.join([t.name for t in tracks])
        n_prev_bars = 0
        notes = []
        for track in tracks:
            t = track.pad_left(n_prev_bars)
            notes += t.get_notes()
            n_prev_bars = t.get_bar_count()
        return Track(program, notes, config, name)

    def add_notes(self, notes : List):
        if len(notes):
            for note in notes:
                if utils.validate_note(self.config, note):
                    self.notes += [note]
                # else:
                #     logging.warn('Invalid note filtered: ' + str(note))
        # else:
            # logging.warn('Empty list of notes entered.')

    def set_program(self, program : int):
        assert 0 <= program < 256, "Invalid program."
        self.program = program
        if program < 128:
            self.inst_family = INSTRUMENT_FAMILIES[program // 8]
            self.is_drum = False
        else:
            self.inst_family = INSTRUMENT_FAMILIES[-1]
            self.is_drum = True    

    def set_name(self, name : str):
        assert isinstance(name, str), "Invalid name."
        self.name = name

    def change_config(self, config : MusicConfig):
        if config is not None and self.config != config:
            for note in self.notes:
                note = utils.update_note_with_config(note, config, self.config)
            self.config = config
            return
        logging.warn('Config unchanged.')

    def pad_left(self, n : int = 0, inplace : bool = False):
        notes = self.get_notes()
        if n > 0:
            for note in notes:
                note.set_metric_attributes(bar=note.bar + n)
        if inplace:
            self.notes = notes
            return
        return Track(self.program, notes, self.get_config(), self.name)

    def find_position_index(self, bar : int, beat : int):
        for i, n in enumerate(self.notes):
            if n.bar >= bar and n.beat >= beat:
                return i
        return -1

    def slice_by_index(self, start : int, end : int, inplace : bool = False):
        if inplace:
            self.notes = self.notes[start:end]
            return
        return Track(self.program, self.get_notes()[start:end], config=self.config, name=self.name)

    def slice_by_beat(self, start : int, end : int, inplace : bool = False):
        s = self.find_position_index(start // self.config.n_bar_steps, start % self.config.n_bar_steps)
        e = self.find_position_index(end // self.config.n_bar_steps, end % self.config.n_bar_steps)
        if s == -1:
            notes = []
        else:
            notes = self.get_notes()[s:]
            if e > -1:
                notes = notes[:e-s] 
        if inplace:
            self.notes = notes
            return
        return Track(self.program, notes, config=self.get_config(), name=self.name)

    def slice_by_bar(self, start : int, end : int, inplace : bool = False):
        s = self.find_position_index(start, 0)
        e = self.find_position_index(end, 0)
        if s == -1:
            notes = []
        else:
            notes = self.get_notes()[s:]
            if e > -1:
                notes = notes[:e-s] 
        if inplace:
            self.notes = notes
            return 
        return Track(self.program, notes, config=self.get_config(), name=self.name)

    def clean(self): 
        # 1. remove identical notes 
        # 2. remove invalid notes
        # 2. sort notes
        self.notes = utils.sort_and_remove_identical_events(self.notes)
        res = []
        for n in self.notes:
            if utils.validate_note(self.config, n):
                res += [n]
        self.notes = res

    def organize(self):
        res = utils.organize_events_by_attr(self.notes, ['bar'])
        for i in res:
            res[i] = utils.organize_events_by_attr(res[i], ['beat'])
            for j in res[i]:
                res[i][j] = Track(self.program, res[i][j], self.config, self.name)
        return res

    def get_bars(self):
        if len(self.notes) == 0:
            return []
        res = utils.organize_events_by_attr(self.notes, ['bar'])
        for i in res:
            res[i] = Track(self.program, res[i], config=self.config, name=self.name)
        return res

    def get_bar_count(self):
        if len(self.notes) > 0:
            return self.notes[-1].bar + 1
        return 0

    def get_instrument(self, family=False):
        return self.inst_family if family else self.program

    def get_notes(self):
        return deepcopy(self.notes)

    def get_config(self):
        return deepcopy(self.config)

    def is_empty(self):
        return len(self.notes) == 0

    def to_midi_instrument(self):
        notes = []
        for note in self.notes:
            s = note.beat*self.config.step + note.bar*self.config.bar_resol
            d = note.duration*self.config.step
            v = self.config.velocity_bins[note.velocity]
            notes += [
                ct.Note(
                    velocity=v, 
                    pitch=note.pitch, 
                    start=s, 
                    end=s + d
                )
            ]
        instrument = ct.Instrument(self.program % 128, self.program > 127, name=self.name)
        instrument.notes = sorted(notes, key=lambda x: x.start)
        max_tick = instrument.notes[-1].end
        return instrument, max_tick

    def to_pianoroll(self, binarize=False):
        roll = np.zeros(shape=(128, self.get_bar_count()*self.config.n_bar_steps))
        for note in self.notes:
            offset = note.bar*self.config.n_bar_steps + note.beat
            roll[note.pitch, offset:offset+note.duration] = 1 if binarize else self.config.velocity_bins[note.velocity]
        return roll

    def to_tokens(self, add_instrument_token=True):
        res = []
        bars = self.organize()
        for bar_idx in bars:
            res += ['Bar']
            for beat_idx in bars[bar_idx]:
                res += ['Beat_' + str(beat_idx)]
                for note in bars[bar_idx][beat_idx].notes:
                    if add_instrument_token:
                        res += ['NoteInstrument_'+str(self.program)]
                    res += note.to_tokens(include_metrics=False)
        return res

    def to_tuples(self, add_instrument_token=True):
        res = []
        for note in self.notes:
            res += [note.to_tuple()]
        res = np.array(res)
        if add_instrument_token: 
            inst_token = np.ones(shape=(res.shape[0], 1)).astype(int) * (self.program + 1)
            res = np.concatenate([res, inst_token], axis=1)
        return res

    def __repr__(self):
        return f'Track(inst_family={self.inst_family}, program={self.program}, name={self.name}, notes={len(self.notes)}, bars={self.get_bar_count()})'
    
    def __len__(self):
        return len(self.notes)

    def __getitem__(self, index):
        return self.notes[index]

    def __hash__(self):
        return hash((self.program, self.config, *self.notes))

    def __eq__(self, other):
        if isinstance(other, Track):
            return self.program == other.program and self.config == other.config and len(self) == len(other) and set(self.notes) == set(other.notes)   
        return False