import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
import librosa.effects


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


def apply_alien_voice(audio_data, sr):
    # Apply an alien voice effect with pitch shifting, time stretching, and volume modulation
    alien_voice_pitch = pitch_shift(audio_data, sr, semitone_shift=12)
    alien_voice_stretch = change_speed(alien_voice_pitch, rate=1.5)
    alien_voice_modulated = increase_volume(alien_voice_stretch, volume_factor=0.8)
    return alien_voice_modulated


def apply_robotic_voice(audio_data, sr):
    # Apply a robotic voice effect
    robotic_voice_pitch = pitch_shift(audio_data, sr, semitone_shift=-3)
    robotic_voice_speed = change_speed(robotic_voice_pitch, rate=1.2)
    return robotic_voice_speed


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


def apply_helium_voice(audio_data, sr):
    # Apply a helium-like voice effect
    helium_voice = pitch_shift(audio_data, sr, semitone_shift=12)
    return helium_voice


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
    return radio_voice


def apply_strong_echo(audio_data, sr, delay_factor=0.7, decay=0.7):
    # Apply a strong echo effect
    echo_audio = np.copy(audio_data)
    delay_samples = int(sr * delay_factor)
    for i in range(delay_samples, len(audio_data)):
        echo_audio[i] += decay * audio_data[i - delay_samples]
    return echo_audio


def apply_cave_voice(audio_data, sr):
    # Apply a cave-like reverb effect
    cave_voice = apply_reverb(audio_data, sr, reverb_amount=0.8)
    return cave_voice


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


def apply_deep_robot_voice(audio_data, sr):
    # Apply a deep robotic voice effect with lower pitch and slight distortion
    robot_voice = pitch_shift(audio_data, sr, semitone_shift=-6)
    robot_voice = np.clip(robot_voice * 2, -1, 1)
    return robot_voice


def apply_fuzzy_voice(audio_data):
    # Apply a fuzzy voice effect by adding distortion
    fuzzy_voice = np.clip(audio_data * 4, -1, 1)
    return fuzzy_voice


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


def apply_squeaky_voice(audio_data, sr):
    # Apply a squeaky-like effect by shifting pitch up significantly
    squeaky_voice = pitch_shift(audio_data, sr, semitone_shift=15)
    return squeaky_voice


def apply_metallic_voice(audio_data, sr):
    # Apply a metallic effect using a combination of pitch shift and slight distortion
    metallic_voice = pitch_shift(audio_data, sr, semitone_shift=-5)
    metallic_voice = np.clip(metallic_voice * 1.5, -1, 1)
    return metallic_voice


def apply_radioactive_voice(audio_data, sr):
    # Apply a radioactive-like effect by combining reverb, pitch shift, and distortion
    radioactive_voice = pitch_shift(audio_data, sr, semitone_shift=-7)
    radioactive_voice = apply_reverb(radioactive_voice, sr, reverb_amount=0.7)
    radioactive_voice = np.clip(radioactive_voice * 2, -1, 1)
    return radioactive_voice


def apply_wobble_voice(audio_data, sr, wobble_frequency=5.0, wobble_width=0.5):
    # Apply a wobble effect using pitch modulation
    t = np.arange(len(audio_data)) / sr
    modulator = wobble_width * np.sin(2 * np.pi * wobble_frequency * t)

    # Create a buffer to store the wobble effect
    wobble_audio = np.zeros_like(audio_data)

    # Apply pitch shift in small chunks to simulate the wobble effect
    chunk_size = 1024
    for i in range(0, len(audio_data), chunk_size):
        end = i + chunk_size
        chunk = audio_data[i:end]
        pitch_shift = modulator[i:end].mean()
        shifted_chunk = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=pitch_shift)
        wobble_audio[i:end] = shifted_chunk

    return wobble_audio


def apply_deep_voice(audio_data, sr):
    # Apply a deep voice effect by shifting pitch down
    deep_voice = pitch_shift(audio_data, sr, semitone_shift=-8)
    return deep_voice


def apply_breathy_voice(audio_data, sr):
    # Apply a breathy effect by adding white noise
    breathy_voice = audio_data * 0.8
    noise = np.random.normal(0, 0.05, len(audio_data))
    breathy_voice += noise
    return breathy_voice


def apply_stadium_voice(audio_data, sr):
    # Apply a stadium effect using a large reverb
    stadium_voice = apply_reverb(audio_data, sr, reverb_amount=0.95)
    return stadium_voice


def apply_stuttering_voice(audio_data, sr, stutter_factor=0.1):
    # Apply a stuttering effect by repeating small segments
    segment_length = int(sr * stutter_factor)
    stuttering_voice = []
    for i in range(0, len(audio_data), segment_length):
        stuttering_voice.extend(audio_data[i:i + segment_length])
        stuttering_voice.extend(audio_data[i:i + int(segment_length / 2)])
    return np.array(stuttering_voice)


def apply_time_warp_voice(audio_data, sr):
    # Apply a time warp effect using dynamic time stretching
    time_warp_voice = librosa.effects.time_stretch(audio_data, rate=0.5)
    time_warp_voice = librosa.effects.time_stretch(time_warp_voice, rate=2.0)
    return time_warp_voice


def apply_ethereal_voice(audio_data, sr):
    # Apply an ethereal effect using pitch shift and reverb
    ethereal_voice = pitch_shift(audio_data, sr, semitone_shift=7)
    ethereal_voice = apply_reverb(ethereal_voice, sr, reverb_amount=0.95)
    return ethereal_voice


def apply_broken_robot_voice(audio_data, sr):
    # Apply a broken robot effect using pitch shift, distortion, and time stretching
    broken_robot_voice = pitch_shift(audio_data, sr, semitone_shift=-4)
    broken_robot_voice = np.clip(broken_robot_voice * 1.5, -1, 1)
    broken_robot_voice = change_speed(broken_robot_voice, rate=0.8)
    return broken_robot_voice


def apply_alien_invasion_voice(audio_data, sr):
    # Apply an alien invasion effect using pitch shift and time stretching
    alien_invasion_voice = pitch_shift(audio_data, sr, semitone_shift=12)
    alien_invasion_voice = change_speed(alien_invasion_voice, rate=0.7)
    return alien_invasion_voice


def apply_slow_down_voice(audio_data, sr):
    # Apply a slow down effect by decreasing the speed
    slow_down_voice = change_speed(audio_data, rate=0.5)
    return slow_down_voice


def apply_giant_voice(audio_data, sr):
    # Apply a giant voice effect by decreasing the pitch significantly
    giant_voice = pitch_shift(audio_data, sr, semitone_shift=-10)
    return giant_voice


def apply_cyborg_voice(audio_data, sr):
    # Apply a cyborg voice effect by combining pitch shifting and time stretching
    cyborg_voice = pitch_shift(audio_data, sr, semitone_shift=-4)
    cyborg_voice = change_speed(cyborg_voice, rate=0.8)
    return cyborg_voice


def apply_vibrato_voice(audio_data, sr, vibrato_frequency=5.0, vibrato_depth=0.5):
    # Apply a vibrato effect using frequency modulation
    t = np.arange(len(audio_data)) / sr
    modulator = vibrato_depth * np.sin(2 * np.pi * vibrato_frequency * t)

    # Create a buffer to store the vibrato effect
    vibrato_audio = np.zeros_like(audio_data)

    # Apply pitch shift in small chunks to simulate the vibrato effect
    chunk_size = 1024
    for i in range(0, len(audio_data), chunk_size):
        end = i + chunk_size
        chunk = audio_data[i:end]
        pitch_shift_amount = modulator[i:end].mean()
        shifted_chunk = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=pitch_shift_amount)
        vibrato_audio[i:end] = shifted_chunk

    return vibrato_audio


def apply_robot_voice_vocoder(audio_data, sr):
    # Apply a robotic effect using a vocoder-like effect
    robot_voice = librosa.effects.harmonic(audio_data)
    robot_voice = pitch_shift(robot_voice, sr, semitone_shift=-3)
    return robot_voice


def apply_radioactive_voice(audio_data, sr):
    # Apply a radioactive-like effect by combining reverb, pitch shift, and distortion
    radioactive_voice = pitch_shift(audio_data, sr, semitone_shift=-7)
    radioactive_voice = apply_reverb(radioactive_voice, sr, reverb_amount=0.7)
    radioactive_voice = np.clip(radioactive_voice * 2, -1, 1)
    return radioactive_voice


def apply_deep_sea_voice(audio_data, sr):
    # Apply a deep sea effect using a combination of low-pass filter and reverb
    sos = butter(10, 200, btype='low', fs=sr, output='sos')
    deep_sea_voice = sosfilt(sos, audio_data)
    deep_sea_voice = apply_reverb(deep_sea_voice, sr, reverb_amount=0.9)
    return deep_sea_voice


def apply_darth_vader_voice(audio_data, sr):
    # Apply a Darth Vader effect by decreasing the pitch and adding reverb
    darth_vader_voice = pitch_shift(audio_data, sr, semitone_shift=-7)
    darth_vader_voice = apply_reverb(darth_vader_voice, sr, reverb_amount=0.7)
    return darth_vader_voice


def apply_radio_announcer_voice(audio_data, sr):
    # Apply a radio announcer effect using equalization and slight reverb
    sos = butter(10, [100, 5000], btype='band', fs=sr, output='sos')
    radio_announcer_voice = sosfilt(sos, audio_data)
    radio_announcer_voice = apply_reverb(radio_announcer_voice, sr, reverb_amount=0.2)
    return radio_announcer_voice


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


def apply_dreamy_voice(audio_data, sr):
    # Apply a dreamy effect using pitch shift and reverb
    dreamy_voice = pitch_shift(audio_data, sr, semitone_shift=2)
    dreamy_voice = apply_reverb(dreamy_voice, sr=sr, reverb_amount=0.7)
    return dreamy_voice


def apply_mad_scientist_voice(audio_data, sr):
    # Apply a mad scientist effect using pitch shift, distortion, and echo
    mad_scientist_voice = pitch_shift(audio_data, sr, semitone_shift=5)
    mad_scientist_voice = np.clip(mad_scientist_voice * 1.3, -1, 1)
    mad_scientist_voice = apply_echo(mad_scientist_voice, sr=sr, delay_factor=0.5, decay=0.6)
    return mad_scientist_voice


def apply_fairy_voice(audio_data, sr):
    # Apply a fairy effect using pitch shift with modulation
    fairy_voice = librosa.effects.pitch_shift(audio_data, sr, n_steps=2)
    t = np.linspace(0, len(fairy_voice) / sr, len(fairy_voice))
    modulator = np.sin(2 * np.pi * 1.5 * t)  # Modulation frequency of 1.5 Hz
    fairy_voice *= 1 + 0.2 * modulator  # Adjust modulation depth as needed
    return fairy_voice


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


# Example usage
if __name__ == "__main__":
    file_path = 'sample_audios/imran_khan_trimmed.mp3'
    audio_data, sr = load_audio(file_path)

    # # Apply voice effects
    shifted_pitch = pitch_shift(audio_data, sr=sr, semitone_shift=2)
    louder = increase_volume(audio_data, volume_factor=1.5)
    echoed = apply_echo(audio_data, sr, delay_factor=0.5, decay=0.5)
    reverb = apply_reverb(audio_data, sr, reverb_amount=0.5)
    girl_voice = apply_girl_voice(audio_data, sr)
    alien_voice = apply_alien_voice(audio_data, sr)
    robotic_voice = apply_robotic_voice(audio_data, sr)
    child_voice = apply_child_voice(audio_data, sr)
    reversed_voice = apply_reversed_voice(audio_data)
    male_voice = apply_male_voice(audio_data, sr)
    demon_voice = apply_demon_voice(audio_data, sr)
    helium_voice = apply_helium_voice(audio_data, sr)
    telephone_voice = apply_telephone_voice(audio_data, sr)
    chipmunk_voice = apply_chipmunk_voice(audio_data, sr)
    slow_motion_voice = apply_slow_motion_voice(audio_data, sr)
    distorted_voice = apply_distorted_voice(audio_data)
    underwater_voice = apply_underwater_voice(audio_data, sr)
    haunted_voice = apply_haunted_voice(audio_data, sr)
    monster_voice = apply_monster_voice(audio_data, sr)
    whisper_voice = apply_whisper_voice(audio_data)
    radio_voice = apply_radio_voice(audio_data, sr)
    strong_echo_voice = apply_strong_echo(audio_data, sr)
    cave_voice = apply_cave_voice(audio_data, sr)
    megaphone_voice = apply_megaphone_voice(audio_data, sr)
    space_voice = apply_space_voice(audio_data, sr)
    deep_robot_voice = apply_deep_robot_voice(audio_data, sr)
    fuzzy_voice = apply_fuzzy_voice(audio_data)
    tremolo_voice = apply_tremolo_voice(audio_data, sr)
    flanger_voice = apply_flanger_voice(audio_data, sr)
    squeky_voice = apply_whisper_voice(audio_data)
    metallic_voice = apply_metallic_voice(audio_data, sr)
    radio_active_voice = apply_radioactive_voice(audio_data, sr)
    wobble_voice = apply_wobble_voice(audio_data, sr)
    deep_voice = apply_deep_voice(audio_data, sr)
    breathy_voice = apply_breathy_voice(audio_data, sr)
    stadium_voice = apply_stadium_voice(audio_data, sr)
    stuttering_voice = apply_stuttering_voice(audio_data, sr)
    time_Warp_voice = apply_time_warp_voice(audio_data, sr)
    etheral_voice = apply_ethereal_voice(audio_data, sr)
    broken_robot_voice = apply_broken_robot_voice(audio_data, sr)
    alien_invasion_voice = apply_alien_invasion_voice(audio_data, sr)
    slow_down_voice = apply_slow_down_voice(audio_data, sr)
    giant_voice = apply_giant_voice(audio_data, sr)
    cyborg_voice = apply_cyborg_voice(audio_data, sr)
    vibrato_voice = apply_vibrato_voice(audio_data, sr)
    robotic_vocoder = apply_robot_voice_vocoder(audio_data, sr)
    deep_sea_voice = apply_deep_sea_voice(audio_data, sr)
    darth_vader_voice = apply_darth_vader_voice(audio_data, sr)
    radio_announcer_voice = apply_radio_announcer_voice(audio_data, sr)
    ghostly_whisper_voice = apply_ghostly_whisper_voice(audio_data, sr)
    cylon_voice = apply_cylon_voice(audio_data, sr)
    evil_witch_voice = apply_evil_witch_voice(audio_data, sr)
    digital_glitch_voice = apply_digital_glitch_voice(audio_data, sr)
    cyberpunk_voice = apply_cyberpunk_voice(audio_data, sr)
    dreamy_voice = apply_dreamy_voice(audio_data, sr)
    mad_scientist_voice = apply_mad_scientist_voice(audio_data, sr)
    fairy_voice = apply_fairy_voice(audio_data, sr)
    cybernetic_voice = apply_cybernetic_voice(audio_data, sr)
    galactic_voice = apply_galactic_voice(audio_data, sr)
    celestial_voice = apply_celestial_voice(audio_data, sr)
    cosmic_voice = apply_cosmic_voice(audio_data, sr)
    mystical_voice = apply_mystical_voice(audio_data, sr)
    enchanted_voice = apply_enchanted_voice(audio_data, sr)
    transcendent_voice = apply_transcendent_voice(audio_data, sr)

    # Save the modified audio files
    save_audio(shifted_pitch, 'results/shifted_pitch.wav', sr)
    save_audio(louder, 'results/louder.wav', sr)
    save_audio(echoed, 'results/echoed.wav', sr)
    save_audio(reverb, 'results/reverb.wav', sr)
    save_audio(girl_voice, 'results/girl_voice.wav', sr)
    save_audio(alien_voice, 'results/alien_voice.wav', sr)
    save_audio(robotic_voice, 'results/robotic_voice.wav', sr)
    save_audio(child_voice, 'results/child_voice.wav', sr)
    save_audio(reversed_voice, 'results/reversed_voice.wav', sr)
    save_audio(male_voice, 'results/male_voice.wav', sr)
    save_audio(demon_voice, 'results/demon_voice.wav', sr)
    save_audio(helium_voice, 'results/helium_voice.wav', sr)
    save_audio(telephone_voice, 'results/telephone_voice.wav', sr)
    save_audio(chipmunk_voice, 'results/chipmunk_voice.wav', sr)
    save_audio(slow_motion_voice, 'results/slow_motion_voice.wav', sr)
    save_audio(distorted_voice, 'results/distorted_voice.wav', sr)
    save_audio(underwater_voice, 'results/underwater_voice.wav', sr)
    save_audio(haunted_voice, 'results/haunted_voice.wav', sr)
    save_audio(monster_voice, 'results/monster_voice.wav', sr)
    save_audio(whisper_voice, 'results/whisper_voice.wav', sr)
    save_audio(radio_voice, 'results/radio_voice.wav', sr)
    save_audio(strong_echo_voice, 'results/strong_echo_voice.wav', sr)
    save_audio(cave_voice, 'results/cave_voice.wav', sr)
    save_audio(megaphone_voice, 'results/megaphone_voice.wav', sr)
    save_audio(space_voice, 'results/space_voice.wav', sr)
    save_audio(deep_robot_voice, 'results/deep_robot_voice.wav', sr)
    save_audio(fuzzy_voice, 'results/fuzzy_voice.wav', sr)
    save_audio(tremolo_voice, 'results/tremolo_voice.wav', sr)
    save_audio(flanger_voice, 'results/flanger_voice.wav', sr)
    save_audio(squeky_voice, 'results/squeky_voice.wav', sr)
    save_audio(metallic_voice, 'results/metallic_voice.wav', sr)
    save_audio(radio_active_voice, 'results/radio_active_voice.wav', sr)
    save_audio(wobble_voice, 'results/wobble_voice.wav', sr)
    save_audio(deep_voice, 'results/deep_voice.wav', sr)
    save_audio(breathy_voice, 'results/breathy_voice.wav', sr)
    save_audio(stadium_voice, 'results/stadium_voice.wav', sr)
    save_audio(stuttering_voice, 'results/stuttering_voice.wav', sr)
    save_audio(time_Warp_voice, 'results/time_Warp_voice.wav', sr)
    save_audio(etheral_voice, 'results/etheral_voice.wav', sr)
    save_audio(broken_robot_voice, 'results/broken_robot_voice.wav', sr)
    save_audio(alien_invasion_voice, 'results/alien_invasion_voice.wav', sr)
    save_audio(slow_down_voice, 'results/slow_down_voice.wav', sr)
    save_audio(giant_voice, 'results/giant_voice.wav', sr)
    save_audio(cyborg_voice, 'results/cyborg_voice.wav', sr)
    save_audio(vibrato_voice, 'results/vibrato_voice.wav', sr)
    save_audio(robotic_vocoder, 'results/robotic_vocoder.wav', sr)
    save_audio(deep_sea_voice, 'results/deep_sea_voice.wav', sr)
    save_audio(darth_vader_voice, 'results/darth_vader_voice.wav', sr)
    save_audio(radio_announcer_voice, 'results/radio_announcer_voice.wav', sr)
    save_audio(ghostly_whisper_voice, 'results/ghostly_whisper_voice.wav', sr)
    save_audio(cylon_voice, 'results/cylon_voice.wav', sr)
    save_audio(evil_witch_voice, 'results/evil_witch_voice.wav', sr)
    save_audio(digital_glitch_voice, 'results/digital_glitch_voice.wav', sr)
    save_audio(cyberpunk_voice, 'results/cyberpunk_voice.wav', sr)
    save_audio(dreamy_voice, 'results/dreamy_voice.wav', sr)
    save_audio(mad_scientist_voice, 'results/mad_scientist_voice.wav', sr)
    save_audio(fairy_voice, 'results/fairy_voice.wav', sr)
    save_audio(cybernetic_voice, 'results/cybernetic_voice.wav', sr)
    save_audio(galactic_voice, 'results/galactic_voice.wav', sr)
    save_audio(celestial_voice, 'results/celestial_voice.wav', sr)
    save_audio(cosmic_voice, 'results/cosmic_voice.wav', sr)
    save_audio(mystical_voice, 'results/mystical_voice.wav', sr)
    save_audio(enchanted_voice, 'results/enchanted_voice.wav', sr)
    save_audio(transcendent_voice, 'results/transcendent_voice.wav', sr)
