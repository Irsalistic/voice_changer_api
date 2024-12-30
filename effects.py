import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
import librosa.effects

bg_effect_strength = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0}


def decrease_volume(audio_data, factor=0.5):
    # Decrease the volume of the audio
    decreased_audio = audio_data * factor
    return decreased_audio


def add_bg_effect(audio_data, sr, thunder_file, effect_start=0, factor=.3):
    thunder_audio, thunder_sr = librosa.load(thunder_file, sr=sr)
    thunder_audio = decrease_volume(thunder_audio, factor)
    assert thunder_sr == sr, "Sample rates do not match"

    start_sample = int(effect_start * sr)
    audio_length = len(audio_data)
    thunder_length = len(thunder_audio)

    if start_sample < 0:
        raise ValueError("thunder_start must be a non-negative value")
    if start_sample + thunder_length < audio_length:
        thunder_audio = np.pad(thunder_audio, (start_sample, audio_length - (start_sample + thunder_length)),
                               'constant')
    else:
        thunder_audio = np.pad(thunder_audio, (start_sample, 0), 'constant')[:audio_length]
    mixed_audio = audio_data + thunder_audio[:audio_length]

    return mixed_audio


def apply_delay(audio_data, sr, delay_time=0.1, feedback=0.4):
    delay_samples = int(sr * delay_time)
    delayed_audio = np.zeros_like(audio_data)
    for i in range(delay_samples, len(audio_data)):
        delayed_audio[i] = audio_data[i] + feedback * delayed_audio[i - delay_samples]
    return delayed_audio


def apply_chorus(audio_data, sr, depth=0.03, delay=0.004, rate=1.3):
    modulator = np.sin(2 * np.pi * rate * np.arange(len(audio_data)) / sr)
    modulator *= depth * sr
    chorus_audio = np.copy(audio_data)
    for i in range(len(audio_data)):
        delay_index = int(i - modulator[i])
        if 0 <= delay_index < len(audio_data):
            chorus_audio[i] += audio_data[delay_index]
    return chorus_audio


def load_audio(file_path):
    # Load the audio file using librosa
    audio_data, sr = librosa.load(file_path, sr=None)
    return audio_data, sr


def save_audio(audio_data, file_path, sr):
    # Save the modified audio to a file
    sf.write(file_path, audio_data, sr)


def pitch_shift(audio_data, sr, semitone_shift):
    # Perform pitch shifting
    shifted_audio = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=semitone_shift)
    return shifted_audio


def increase_volume(audio_data, volume_factor):
    # Increase the volume of the audio
    audio_data *= volume_factor
    return audio_data


def change_speed(audio_data, rate):
    # Change the speed of the audio
    sped_audio = librosa.effects.time_stretch(audio_data, rate=rate)
    return sped_audio


def apply_echo(audio_data, sr, delay_factor=0.5, decay=0.5):
    # Apply echo effect using repetition with decay
    echo_audio = np.copy(audio_data)
    delay_samples = int(sr * delay_factor)
    for i in range(delay_samples, len(audio_data)):
        echo_audio[i] += decay * audio_data[i - delay_samples]
    return echo_audio


def apply_reverb(audio_data, sr, reverb_amount=0.7):
    # Apply a reverb effect
    reverb_data = librosa.effects.preemphasis(audio_data)
    reverb_data = np.clip(reverb_data * reverb_amount, -1, 1)
    return reverb_data


def apply_girl_voice(audio_data, sr):
    # Apply pitch shifting to increase the pitch
    girl_voice = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=12)  # Increase the pitch by 12 semitones

    # Apply time stretching for a more natural sound
    girl_voice = librosa.effects.time_stretch(girl_voice, rate=1.2)  # Increase the duration by 20%

    # Apply fade in/out for smoothness
    fade_length = int(0.03 * sr)  # Length of fade in samples
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    girl_voice[:fade_length] *= fade_in
    girl_voice[-fade_length:] *= fade_out

    return girl_voice


def apply_child_voice(audio_data, sr):
    # Apply a child-like voice effect
    child_voice = pitch_shift(audio_data, sr, semitone_shift=7)
    return child_voice


def apply_reversed_voice(audio_data):
    # Apply a reversed voice effect
    reversed_voice = audio_data[::-1]
    return reversed_voice


def apply_male_voice(audio_data, sr):
    # Apply a lower-pitched effect for a male voice
    male_voice = pitch_shift(audio_data, sr, semitone_shift=-3)
    return male_voice


def apply_demon_voice(audio_data, sr):
    # Apply a demon-like voice effect
    demon_voice = pitch_shift(audio_data, sr, semitone_shift=-12)
    return demon_voice


def apply_telephone_voice(audio_data, sr):
    # Apply a telephone-like voice effect by applying a bandpass filter
    lowcut = 300.0
    highcut = 3400.0
    sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    telephone_voice = sosfilt(sos, audio_data)
    return telephone_voice


def apply_chipmunk_voice(audio_data, sr):
    # Apply a chipmunk-like voice effect
    chipmunk_voice = pitch_shift(audio_data, sr, semitone_shift=15)
    return chipmunk_voice


def apply_slow_motion_voice(audio_data, sr):
    # Apply a slow-motion voice effect
    slow_voice_pitch = pitch_shift(audio_data, sr, semitone_shift=-5)
    slow_voice_speed = change_speed(slow_voice_pitch, rate=0.5)
    return slow_voice_speed


def apply_distorted_voice(audio_data):
    # Apply a distorted voice effect
    distorted_voice = np.clip(audio_data * 10, -1, 1)
    return distorted_voice


def apply_underwater_voice(audio_data, sr):
    # Apply an underwater voice effect
    lowcut = 300.0
    highcut = 600.0
    sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    underwater_voice = sosfilt(sos, audio_data)
    return underwater_voice


def apply_haunted_voice(audio_data, sr):
    # Apply a haunted voice effect
    haunted_voice = apply_reverb(audio_data, sr, reverb_amount=0.9)
    haunted_voice = apply_echo(haunted_voice, sr, delay_factor=0.3, decay=0.8)
    return haunted_voice


def apply_monster_voice(audio_data, sr):
    # Apply a monster-like voice effect
    monster_voice = pitch_shift(audio_data, sr, semitone_shift=-9)
    monster_voice = apply_distorted_voice(monster_voice)
    return monster_voice


def apply_whisper_voice(audio_data):
    # Apply a whisper-like effect by reducing volume and adding white noise
    whisper_audio = audio_data * 0.2
    noise = np.random.normal(0, 0.02, len(audio_data))
    whisper_audio += noise
    return whisper_audio


def apply_radio_voice(audio_data, sr):
    # Apply a radio-like effect by using a bandpass filter and adding noise
    lowcut = 300.0
    highcut = 3000.0
    sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    radio_voice = sosfilt(sos, audio_data)
    noise = np.random.normal(0, 0.01, len(audio_data))
    radio_voice += noise
    effect_file = f'effects_sounds/radio.wav'
    mixed_audio = add_bg_effect(radio_voice, sr, effect_file, effect_start=0)
    return mixed_audio


def apply_strong_echo(audio_data, sr, delay_factor=0.7, decay=0.7):
    # Apply a strong echo effect
    echo_audio = np.copy(audio_data)
    delay_samples = int(sr * delay_factor)
    for i in range(delay_samples, len(audio_data)):
        echo_audio[i] += decay * audio_data[i - delay_samples]
    return echo_audio


def apply_megaphone_voice(audio_data, sr):
    # Apply a megaphone-like effect with bandpass filtering and distortion
    lowcut = 500.0
    highcut = 5000.0
    sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    megaphone_voice = sosfilt(sos, audio_data)
    megaphone_voice = np.clip(megaphone_voice * 5, -1, 1)
    return megaphone_voice


def apply_space_voice(audio_data, sr):
    # Apply a space-like effect using reverb and echo
    space_voice = apply_reverb(audio_data, sr, reverb_amount=0.9)
    space_voice = apply_echo(space_voice, sr, delay_factor=0.5, decay=0.6)
    return space_voice


def apply_deep_voice(audio_data, sr):
    # Apply a deep robotic voice effect with lower pitch and slight distortion
    robot_voice = pitch_shift(audio_data, sr, semitone_shift=-6)
    robot_voice = np.clip(robot_voice * 2, -1, 1)
    return robot_voice


def apply_tremolo_voice(audio_data, sr):
    # Apply a tremolo effect by modulating the amplitude
    t = np.arange(len(audio_data)) / sr
    tremolo = 0.5 * (1.0 + np.sin(2.0 * np.pi * 5.0 * t))
    tremolo_voice = audio_data * tremolo
    return tremolo_voice


def apply_flanger_voice(audio_data, sr):
    # Apply a flanger effect
    flanger_audio = np.copy(audio_data)
    max_delay = int(0.003 * sr)  # 3 ms delay
    delay_samples = np.arange(0, max_delay)
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 0.25 * np.arange(len(audio_data)) / sr))
    for i in range(max_delay, len(audio_data)):
        delay = int(modulation[i] * max_delay)
        flanger_audio[i] += 0.5 * audio_data[i - delay]
    return flanger_audio


def apply_stuttering_voice(audio_data, sr, stutter_factor=0.1):
    # Apply a stuttering effect by repeating small segments
    segment_length = int(sr * stutter_factor)
    stuttering_voice = []
    for i in range(0, len(audio_data), segment_length):
        stuttering_voice.extend(audio_data[i:i + segment_length])
        stuttering_voice.extend(audio_data[i:i + int(segment_length / 2)])
    return np.array(stuttering_voice)


def apply_broken_robot_voice(audio_data, sr):
    # Apply a broken robot effect using pitch shift, distortion, and time stretching
    broken_robot_voice = pitch_shift(audio_data, sr, semitone_shift=-4)
    broken_robot_voice = np.clip(broken_robot_voice * 1.5, -1, 1)
    broken_robot_voice = change_speed(broken_robot_voice, rate=0.8)
    return broken_robot_voice


def apply_alien_voice(audio_data, sr):
    # Apply an alien invasion effect using pitch shift and time stretching
    alien_invasion_voice = pitch_shift(audio_data, sr, semitone_shift=12)
    alien_invasion_voice = change_speed(alien_invasion_voice, rate=0.7)
    effect_file = f'effects_sounds/robotwav.wav'
    mixed_audio = add_bg_effect(alien_invasion_voice, sr, effect_file, effect_start=0)
    return mixed_audio


def apply_slow_down_voice(audio_data, sr):
    # Apply a slow down effect by decreasing the speed
    slow_down_voice = change_speed(audio_data, rate=0.5)
    return slow_down_voice


def apply_cyborg_voice(audio_data, sr):
    # Apply a cyborg voice effect by combining pitch shifting and time stretching
    cyborg_voice = pitch_shift(audio_data, sr, semitone_shift=-4)
    cyborg_voice = change_speed(cyborg_voice, rate=0.8)
    return cyborg_voice


def apply_robot_voice_vocoder(audio_data, sr):
    # Apply a robotic effect using a vocoder-like effect
    robot_voice = librosa.effects.harmonic(audio_data)
    robot_voice = pitch_shift(robot_voice, sr, semitone_shift=-3)
    return robot_voice


def apply_darth_vader_voice(audio_data, sr):
    # Apply a Darth Vader effect by decreasing the pitch and adding reverb
    darth_vader_voice = pitch_shift(audio_data, sr, semitone_shift=-7)
    darth_vader_voice = apply_reverb(darth_vader_voice, sr, reverb_amount=0.7)
    return darth_vader_voice


def apply_ghostly_whisper_voice(audio_data, sr):
    # Apply a ghostly whisper effect by decreasing the pitch and applying high reverb
    ghostly_whisper_voice = pitch_shift(audio_data, sr, semitone_shift=-5)
    ghostly_whisper_voice = apply_reverb(ghostly_whisper_voice, sr, reverb_amount=0.95)
    return ghostly_whisper_voice


def apply_cylon_voice(audio_data, sr):
    # Apply a Cylon effect by combining pitch shift, time stretch, and ring modulation
    cylon_voice = pitch_shift(audio_data, sr, semitone_shift=-6)
    cylon_voice = change_speed(cylon_voice, rate=0.8)
    t = np.arange(len(cylon_voice)) / sr
    modulator = np.sin(2 * np.pi * 30 * t)  # 30 Hz ring modulation
    cylon_voice = cylon_voice * modulator
    return cylon_voice


def apply_evil_witch_voice(audio_data, sr):
    # Apply an evil witch effect using pitch shift and echo
    evil_witch_voice = pitch_shift(audio_data, sr, semitone_shift=-3)
    evil_witch_voice = apply_echo(evil_witch_voice, sr=sr, delay_factor=0.3, decay=0.6)
    return evil_witch_voice


def apply_digital_glitch_voice(audio_data, sr):
    # Apply a digital glitch effect using random noise injection
    glitch_factor = 0.05  # Adjust glitch intensity as needed
    glitched_audio = audio_data + glitch_factor * np.random.normal(size=len(audio_data))
    return glitched_audio


def apply_cyberpunk_voice(audio_data, sr):
    # Apply a cyberpunk effect using pitch shift, distortion, and echo
    cyberpunk_voice = pitch_shift(audio_data, sr, semitone_shift=4)
    cyberpunk_voice = np.clip(cyberpunk_voice * 1.5, -1, 1)
    cyberpunk_voice = apply_echo(cyberpunk_voice, sr=sr, delay_factor=0.4, decay=0.5)
    return cyberpunk_voice


def apply_mad_scientist_voice(audio_data, sr):
    # Apply a mad scientist effect using pitch shift, distortion, and echo
    mad_scientist_voice = pitch_shift(audio_data, sr, semitone_shift=5)
    mad_scientist_voice = np.clip(mad_scientist_voice * 1.3, -1, 1)
    mad_scientist_voice = apply_echo(mad_scientist_voice, sr=sr, delay_factor=0.5, decay=0.6)
    return mad_scientist_voice


def apply_cybernetic_voice(audio_data, sr):
    # Apply a cybernetic effect using pitch shift, distortion, and ring modulation
    cybernetic_voice = pitch_shift(audio_data, sr, semitone_shift=3)
    cybernetic_voice = np.clip(cybernetic_voice * 1.4, -1, 1)
    t = np.arange(len(cybernetic_voice)) / sr
    modulator = np.sin(2 * np.pi * 20 * t)  # 20 Hz ring modulation
    cybernetic_voice = cybernetic_voice * modulator
    return cybernetic_voice


def apply_galactic_voice(audio_data, sr):
    # Apply a galactic effect using pitch shift, echo, and reverb
    galactic_voice = pitch_shift(audio_data, sr, semitone_shift=3)
    galactic_voice = apply_echo(galactic_voice, sr=sr, delay_factor=0.5, decay=0.5)
    galactic_voice = apply_reverb(galactic_voice, sr, reverb_amount=0.8)
    return galactic_voice


def apply_celestial_voice(audio_data, sr):
    # Apply a celestial effect using chorus and reverb
    celestial_voice = apply_chorus(audio_data, sr=sr, depth=0.5)
    celestial_voice = apply_reverb(celestial_voice, sr, reverb_amount=0.6)
    return celestial_voice


def apply_cosmic_voice(audio_data, sr):
    # Apply a cosmic effect using pitch shift, echo, and reverb
    cosmic_voice = pitch_shift(audio_data, sr, semitone_shift=5)
    cosmic_voice = apply_echo(cosmic_voice, sr=sr, delay_factor=0.3, decay=0.5)
    cosmic_voice = apply_reverb(cosmic_voice, sr, reverb_amount=0.8)
    return cosmic_voice


def apply_mystical_voice(audio_data, sr):
    # Apply a mystical effect using pitch shift, chorus, and delay
    mystical_voice = pitch_shift(audio_data, sr, semitone_shift=3)
    mystical_voice = apply_chorus(mystical_voice, sr, depth=0.02, delay=0.003, rate=1.2)
    mystical_voice = apply_delay(mystical_voice, sr, delay_time=0.05, feedback=0.3)
    return mystical_voice


def apply_enchanted_voice(audio_data, sr):
    # Apply an enchanted effect using pitch shift, chorus, and reverb
    enchanted_voice = pitch_shift(audio_data, sr, semitone_shift=4)
    enchanted_voice = apply_chorus(enchanted_voice, sr, depth=0.03, delay=0.004, rate=1.3)
    enchanted_voice = apply_reverb(enchanted_voice, sr, reverb_amount=0.7)
    return enchanted_voice


def apply_transcendent_voice(audio_data, sr):
    # Apply a transcendent effect using pitch shift, reverse, and delay
    transcendent_voice = pitch_shift(audio_data, sr, semitone_shift=6)
    transcendent_voice = apply_reversed_voice(transcendent_voice)
    transcendent_voice = apply_delay(transcendent_voice, sr, delay_time=0.1, feedback=0.5)
    return transcendent_voice


def apply_whistle_voice(audio_data, sr):
    whistle_tone = np.sin(2 * np.pi * 1500 * np.arange(len(audio_data)) / sr)
    return audio_data + 0.2 * whistle_tone


def apply_synthetic_voice(audio_data, sr):
    synthetic_voice = librosa.effects.time_stretch(audio_data, 1.3)
    synthetic_voice = apply_echo(synthetic_voice, sr, delay_factor=0.2, decay=0.5)
    return synthetic_voice


def apply_gargling_voice(audio_data, sr):
    modulated_signal = np.sin(8 * np.pi * 10 * np.arange(len(audio_data)) / sr)
    gargling_voice = audio_data * modulated_signal
    return gargling_voice


def apply_warrior_shout_voice(audio_data, sr):
    warrior_shout_voice = librosa.effects.pitch_shift(audio_data, sr, n_steps=-3)
    warrior_shout_voice = apply_reverb(warrior_shout_voice, sr, reverb_amount=0.5)
    return warrior_shout_voice


def apply_effect(audio_data, sr, effect_name, start_effect, factor):
    decreased_audio = decrease_volume(audio_data, factor=.8)

    # Load the appropriate effect file
    effect_file = f'effects_sounds/{effect_name}.wav'
    factor = bg_effect_strength.get(factor)
    mixed_audio = add_bg_effect(decreased_audio, sr, effect_file, effect_start=start_effect, factor=factor)
    return mixed_audio


effect_functions = {
    "alien": apply_alien_voice,
    "delay": apply_delay,
    "chorus": apply_chorus,
    "pitch_shift": pitch_shift,
    "increase_volume": increase_volume,
    "change_speed": change_speed,
    "echo": apply_echo,
    "reverb": apply_reverb,
    "girl": apply_girl_voice,
    "child": apply_child_voice,
    "reversed": apply_reversed_voice,
    "male": apply_male_voice,
    "demon": apply_demon_voice,
    "telephone": apply_telephone_voice,
    "chipmunk": apply_chipmunk_voice,
    "slow_motion": apply_slow_motion_voice,
    "distorted": apply_distorted_voice,
    "underwater": apply_underwater_voice,
    "haunted": apply_haunted_voice,
    "monster": apply_monster_voice,
    "whisper": apply_whisper_voice,
    "radio": apply_radio_voice,
    "strong_echo": apply_strong_echo,
    "megaphone": apply_megaphone_voice,
    "space": apply_space_voice,
    "deep": apply_deep_voice,
    "tremolo": apply_tremolo_voice,
    "flanger": apply_flanger_voice,
    "stuttering": apply_stuttering_voice,
    "broken_robot": apply_broken_robot_voice,
    "slow_down": apply_slow_down_voice,
    "cyborg": apply_cyborg_voice,
    "robot": apply_robot_voice_vocoder,
    "darth_vader": apply_darth_vader_voice,
    "ghostly_whisper": apply_ghostly_whisper_voice,
    "cylon": apply_cylon_voice,
    "witch": apply_evil_witch_voice,
    "glitch": apply_digital_glitch_voice,
    "cyberpune": apply_cyberpunk_voice,
    "mad_scientist": apply_mad_scientist_voice,
    "cybernetic": apply_cybernetic_voice,
    "galactic": apply_galactic_voice,
    "celestial": apply_celestial_voice,
    "cosmic": apply_cosmic_voice,
    "mystical": apply_mystical_voice,
    "enchanted": apply_enchanted_voice,
    "transcendent": apply_transcendent_voice,
    "whistle": apply_whistle_voice,
    "synthetic": apply_synthetic_voice,
    "gargling": apply_gargling_voice,
    "warrior": apply_warrior_shout_voice,
}
