# Voice Changer API

This FastAPI application provides a variety of voice and sound effects for uploaded audio files. Users can upload audio files and apply different effects to them, receiving the processed audio file in return.

## Features

- Apply different voice effects like pitch shifting, speed change, echo, reverb, etc.
- Supports background sound effects and overlays.
- Returns processed audio files without saving them to disk.

## Endpoints

1.
### `/voice_changer`

**POST**: Upload an audio file and apply a voice effect.

#### Request

- `audio_file`: The audio file to be processed.
- `category_name`: The name of the effect to be applied. Available effects are listed in `effects.py`.

#### Response

- Returns the processed audio file as a `audio/wav` stream.

#### Example

```bash
curl --location 'http://127.0.0.1:8000/voice_changer' \
--form 'audio_file=@"/C:/Users/Gaming PC/PycharmProjects/voice_changer_API/sample_audios/imran_khan_trimmed.mp3"' \
--form 'category_name="alien"' 
```

2.
### `/voice_effect`

**POST**: Upload an audio file and apply a background effect.

#### Request

- `audio_file`: The audio file to be processed.
- `effect_name`: The name of the background effect to be applied.
- `effect_start`: The start time of the effect in seconds.
- `effect_strength`: The intensity of the effect.

#### Response

- Returns the processed audio file as a `audio/wav` stream.

#### Example

```bash
curl --location 'http://127.0.0.1:8000/voice_effect' \
--form 'audio_file=@"/C:/Users/Gaming PC/PycharmProjects/voice_changer_API/sample_audios/hitler.wav"' \
--form 'effect_name="clapping"' \
--form 'effect_start="0"' \
--form 'effect_strength="2"'"
"
```


