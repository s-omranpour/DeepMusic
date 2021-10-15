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
            elif tok.startswith('NotePitch'):
                assert tokens[idx+3].startswith('NoteInstrument')
                program = int(tokens[idx+3][15:])
                if program not in tracks:
                    tracks[program] = []
                tracks[program] += [NoteEvent.from_tokens([prev_pos_tok] + tokens[idx:idx+3], bar=nb)] 
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
        tempos = utils.sort_and_remove_identical_events(utils.flatten([m.tempos for m in musics]))
        chords = utils.sort_and_remove_identical_events(utils.flatten([m.chords for m in musics]))
        res = Music(tracks, tempos, chords, config, name='Merged from '+', '.join([m.name for m in musics]))
        if merge_similar_programs:
            programs = set([t.program for t in res.tracks])
            for program in programs:
                indices = [idx for idx in range(len(res.tracks)) if res.tracks[idx].program == program]
                res.merge_tracks(indices)
        return res      

    @staticmethod
    def concatenate(musics : List, merge_similar_programs=True):
        """
            if merge == True:
                merge tracks with same program and different names
            else:
                merge tracks with same program and name
        """
        assert len(musics) > 0, "Not enough musics to concatenate."
        ## check for config consistency
        configs = list(set([m.config for m in musics]))
        assert len(configs) == 1, "Inconsistancy among inserted musics' configs."
        config = configs[0]

        all_tracks = musics[0].get_tracks()
        tempos = musics[0].tempos
        chords = musics[0].chords
        n_bar = musics[0].get_bar_count()
        for music in musics[1:]:
            music.pad_left(n_bar)
            n_bar += music.get_bar_count()
            all_tracks += music.get_tracks()
            tempos += music.tempos()
            chords += music.chords()
        tempos = utils.sort_and_remove_identical_events(utils.flatten([m.tempos for m in musics]))
        chords = utils.sort_and_remove_identical_events(utils.flatten([m.chords for m in musics]))
        res = Music(all_tracks, tempos, chords, config, name='Concatented from '+', '.join([m.name for m in musics]))
        if merge_similar_programs:
            programs = set([t.program for t in res.tracks])
            for program in programs:
                indices = [idx for idx in range(len(res.tracks)) if res.tracks[idx].program == program]
                res.merge_tracks(indices)
        return res

    def change_config(self, config : MusicConfig):
        for t in self.tracks:
            t.change_config(config)

        for idx,tempo in enumerate(self.tempos):
            tempo.tempo = np.argmin(np.abs(config.tempo_bins - self.config.tempo_bins[tempo.tempo]))
            self.tempos[idx] = utils.update_metric_attributes_with_config(tempo, config, self.config)

        for idx,chord in enumerate(self.chords):
            self.chords[idx] = utils.update_metric_attributes_with_config(chord, config, self.config)
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

    def merge_tracks(self, indices):
        self.tracks += [Track.merge_tracks([self.tracks[idx] for idx in indices])]
        self.remove_tracks(indices=indices)

    def pad_left(self, n : int):
        for track in self.tracks:
            track.pad_left(n)
        for tempo in self.tempos:
            tempo.set_metric_attributes(bar=tempo.bar + n)
        for chord in self.chords:
            chord.set_metric_attributes(bar=chord.bar + n)

    def remove_starting_silent_bars(self):
        min_bars = [t.notes[0].bar for t in self.tracks]
        if len(self.tempos):
            min_bars += [self.tempos[0].bar]
        if len(self.chords):
            min_bars += [self.chords[0].bar]
        offset = min(min_bars)
        for t in self.tracks:
            for note in t.notes:
                note.set_metric_attributes(bar=note.bar - offset)
        for t in self.tempos:
            t.set_metric_attributes(bar=t.bar - offset)
        for c in self.chords:
            c.set_metric_attributes(bar=c.bar - offset)

    def remove_tracks(self, indices:List=None, programs:List=None, inst_families:List=None):
        if indices is not None:
            selected_tracks = [self.tracks[idx] for idx in indices]
            for t in selected_tracks:
                self.tracks.remove(t)
        if programs is not None:
            for t in self.tracks:
                if t.program in programs:
                    self.tracks.remove(t)
        if inst_families is not None:
            for t in self.tracks:
                if t.inst_family in inst_families:
                    self.tracks.remove(t)

    def keep_tracks(self, indices:List=None, programs:List=None, inst_families:List=None):
        tracks = []
        if indices is not None:
            for idx in indices:
                tracks += [self.tracks[idx]]

        if programs is not None:
            for t in self.tracks:
                if t.program in programs:
                    tracks += [t]

        if inst_families is not None:
            for t in self.tracks:
                if t.inst_family in inst_families:
                    tracks += [t]
        self.tracks = tracks

    def find_beat_index(self, events, beat):
        for i, e in enumerate(events):
            if beat <= e.bar*self.config.n_bar_steps + e.beat:
                return i
        return -1

    def slice_by_beat(self, start, end):
        tracks = [t.slice_by_beat(start, end) for t in self.tracks]
        tracks = list(filter(lambda x: not x.is_empty(), tracks))

        s = self.find_beat_index(self.tempos, start)
        e = self.find_beat_index(self.tempos, end + 1)
        if s == -1:
            tempos = []
        else:
            tempos = self.tempos[s:]
            if e > -1:
                tempos = tempos[:e]
        
        s = self.find_beat_index(self.chords, start)
        e = self.find_beat_index(self.chords, end + 1)
        if s == -1:
            chords = []
        else:
            chords = self.chords[s:]
            if e > -1:
                chords = chords[:e]
        return Music(tracks, tempos, chords, self.config, name=self.name + f' - sliced from beat {start} to {end}')

    def slice_by_bar(self, start : int, end : int):
        tracks = [t.slice_by_bar(start, end) for t in self.tracks]
        tracks = list(filter(lambda x: not x.is_empty(), tracks))

        s = self.find_beat_index(self.tempos, start*self.config.n_bar_steps)
        e = self.find_beat_index(self.tempos, (end + 1)*self.config.n_bar_steps)
        if s == -1:
            tempos = []
        else:
            tempos = self.tempos[s:]
            if e > -1:
                tempos = tempos[:e]
        
        s = self.find_beat_index(self.chords, start*self.config.n_bar_steps)
        e = self.find_beat_index(self.chords, (end + 1)*self.config.n_bar_steps)
        if s == -1:
            chords = []
        else:
            chords = self.chords[s:]
            if e > -1:
                chords = chords[:e]
        return Music(tracks, tempos, chords, self.config, name=self.name + f' - sliced from bar {start} to {end}')

    def get_tracks(self):
        return deepcopy(self.tracks)

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
                )
                res[i].remove_starting_silent_bars()
            else:
                del res[i]
        return res

    def get_bar_count(self):
        n_bars = [t.get_bar_count() for t in self.tracks]
        if len(self.tempos):
            n_bars += [self.tempos[-1].bar]
        if len(self.chords):
            n_bars += [self.chords[-1].bar]
        return max(n_bars)

    def get_num_notes(self):
        return sum([len(t) for t in self.tracks])

    def get_instruments(self, family=False):
        return list(set([t.inst_family if family else t.program for t in self.tracks]))

    def to_tokens(self, return_indices=True, add_eos=False):
        res = []
        tracks = [t.organize() for t in self.tracks]
        tempos = utils.organize_events_by_attr(self.tempos, ['bar', 'beat'])
        chords = utils.organize_events_by_attr(self.chords, ['bar', 'beat'])
        for bar_idx in range(self.get_bar_count()):
            res += ['Bar']
            for beat in range(self.const.n_bar_steps):
                if (bar_idx,beat) in tempos:
                    res += ['Beat'+str(beat), 'Tempo'+str(tempos[(bar_idx, beat)][-1].tempo)]
                if (bar_idx,beat) in chords:
                    res += ['Beat'+str(beat), 'Chord'+str(chords[(bar_idx, beat)][-1].chord)]
                for track in tracks:
                    if bar_idx in track and beat in track[bar_idx]:
                        res += track[bar_idx][beat].to_tokens(add_instrument_token=True)[1:] ## each track starts with a bar
        res = utils.remove_duplicate_beats_from_tokens(res)
        if add_eos:
            res += [self.const.special_tokens[1]]
        return self.const.encode(res) if return_indices else res

    def to_tuples(self):
        res = []
        tracks = [t.organize() for t in self.tracks]
        tempos = utils.organize_events_by_attr(self.tempos, ['bar', 'beat'])
        chords = utils.organize_events_by_attr(self.chords, ['bar', 'beat'])
        for bar_idx in range(self.get_bar_count()):
            for beat in range(self.const.n_bar_steps):
                res += [[bar_idx, beat] + [0]*6]
                has_tempo_or_chord = False
                if (bar_idx, beat) in tempos:
                    has_tempo_or_chord = True
                    res[-1][2] = tempos[(bar_idx, beat)].tempo + 1 ## in tuples 0 = ignore
                if (bar_idx, beat) in chords:
                    has_tempo_or_chord = True
                    res[-1][3] = chords[(bar_idx, beat)].chord + 1
                if not has_tempo_or_chord:
                    res = res[:-1] ## remove last tuple if there was no tempos or chords
                has_notes = False
                for track in tracks:
                    if bar_idx in track and beat in track[bar_idx]:
                        has_notes = True
                        res += [track[bar_idx][beat].to_tuples(add_instrument_token=True)]
                if not has_notes and not has_tempo_or_chord:
                    res += [[bar_idx, beat] + [0]*6]
        return np.array(res)

    def to_pianoroll(self, binarize=False, add_tempo_chord=False):
        return dict(
            zip(
                [t.program for t in self.tracks],
                [t.to_pianoroll(binarize=binarize, add_tempo_chord=add_tempo_chord) for t in self.tracks]
            )
        )

    def to_midi(self, output_path : str = None):
        instruments, max_ticks = list(zip(*[t.to_midi_instrument() for t in self.tracks]))
        midi = MidiFile()
        midi.ticks_per_beat = self.const.tick_resol
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
        tracks ='\n    '.join([str(t) for t in self.tracks])
        return f"Music(\n  name={self.name},\n  tracks=[\n    {tracks}\n  ]\n)"

    def __hash__(self):
        return hash((self.config, *self.tempos, *self.chords, *self.tracks))

    def __eq__(self, o: object):
        if isinstance(o, Music):
            return self.config == o.config and set(self.tempos) == set(o.tempos) and set(self.chords) == set(o.chords) and set(self.tracks) == set(self.tracks)


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
    def merge_tracks(tracks : List):
        configs = list(set([t.get_config() for t in tracks]))
        assert len(configs) == 1, "Inconsistent MusicConfig among tracks."
        config = configs[0]
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Merged :' + ', '.join([t.name for t in tracks])
        return Track(program, utils.flatten([t.get_notes() for t in tracks]), config, name)
                
    @staticmethod
    def concatente(tracks: List):
        configs = list(set([t.get_config() for t in tracks]))
        assert len(configs) == 1, "Inconsistent MusicConfig among tracks."
        
        config = configs[0]
        program = Counter([t.program for t in tracks]).most_common(1)[0][0]
        name = 'Concatenated :' + ', '.join([t.name for t in tracks])
        n_prev_bars = [0] + [t.get_bar_count() for t in tracks[:-1]]
        notes = []
        for nb, track in zip(n_prev_bars ,tracks):
            t = deepcopy(track)
            t.pad_left(nb)
            notes += t.get_notes()
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

    def pad_left(self, n : int = 0):
        if n > 0:
            for note in self.notes:
                note.set_metric_attributes(bar=note.bar + n)
        return bool(n)

    def find_beat_index(self, beat : int):
        for i, n in enumerate(self.notes):
            if beat <= n.bar*self.config.n_bar_steps + n.beat:
                return i
        return -1

    def slice_by_index(self, start : int, end : int):
        return Track(self.program, self.notes[start:end], config=self.config, name=self.name + f' - sliced from index {start} to {end}')

    def slice_by_beat(self, start : int, end : int):
        s = self.find_beat_index(start)
        e = self.find_beat_index(end + 1)
        if s == -1:
            notes = []
        else:
            notes = self.notes[s:]
            if e > -1:
                notes = notes[:e] 
        return Track(self.program, notes, config=self.config, name=self.name + f' - sliced from beat {start} to {end}')

    def slice_by_bar(self, start : int, end : int):
        s = self.find_beat_index(start*self.config.n_bar_steps)
        e = self.find_beat_index((end + 1)*self.config.n_bar_steps)
        if s == -1:
            notes = []
        else:
            notes = self.notes[s:]
            if e > -1:
                notes = notes[:e] 
        return Track(self.program, notes, config=self.config, name=self.name + f' - sliced from bar {start} to {end}')

    def clean(self): 
        # 1. remove identical notes 
        # 2. sort notes
        self.notes = utils.sort_and_remove_identical_events(self.notes)

    def organize(self):
        res = utils.organize_events_by_attr(self.notes, ['bar'])
        for i in res:
            res[i] = utils.organize_events_by_attr(res[i])
            for j in res[i]:
                res[i][j] = Track(self.program, res[i][j], self.config, f'Bar No. {i} - Beat No. {j}')
        return res

    def get_bars(self):
        if len(self.notes) == 0:
            return []
        res = utils.organize_events_by_attr(self.notes, ['bar'])
        for i in res:
            res[i] = Track(self.program, res[i], config=self.config, name=self.name + ' - Bar No. ' + str(i))
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
        return self.config

    def get_name(self):
        return self.name

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

    def to_pianoroll(self, binarize=False, add_tempo_chord=True):
        roll = np.zeros(shape=(128, self.get_bar_count()*self.config.n_bar_steps))
        for note in self.notes:
            offset = note.bar*self.config.n_bar_steps + note.beat
            roll[note.pitch, offset:offset+note.duration] = 1 if binarize else self.config.velocity_bins[note.velocity]
        return roll

    def to_tokens(self, add_instrument_token=True):
        res = []
        bars = self.organize()
        for bar in bars:
            res += ['Bar']
            for pos in bar:
                res += ['Beat' + str(pos)]
                for note in bar[pos].notes:
                    res += note.to_tokens(include_metrics=False)
                    if add_instrument_token:
                        res += ['NoteInstrument_'+str(self.program)]
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
        return f'Track(inst_family={self.inst_family}, program={self.program}, name={self.name})'
    
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