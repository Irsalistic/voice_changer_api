import io
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.responses import FileResponse, StreamingResponse
from effects import *

from effects_sounds import *

app = FastAPI()


@app.post("/voice_changer")
async def upload_audio(audio_file: UploadFile = File(...), category_name: str = Form(...)):
    available_categories = [" ,".join(effect_functions.keys())]

    if category_name not in effect_functions:
        raise HTTPException(status_code=400,
                            detail=f"Invalid category. Available categories are: {available_categories}")
    try:
        # Read the uploaded audio into memory
        audio_bytes = await audio_file.read()
        temp_file_path = "audio_file.wav"
        # Open the file in binary write mode
        with open(temp_file_path, "wb") as audio_file:
            # Write the audio bytes to the file
            audio_file.write(audio_bytes)
        # Load the audio from memory
        audio_data, sr = load_audio(temp_file_path)
        # Apply the chosen effect
        effect_function = effect_functions[category_name]
        processed_audio = effect_function(audio_data, sr)
        # Convert the processed audio to bytes
        output_bytes = io.BytesIO()
        sf.write(output_bytes, processed_audio, sr, format='wav')
        output_bytes.seek(0)
        # Return the processed audio as a streaming response
        return StreamingResponse(output_bytes, media_type="audio/wav")
    except HTTPException:
        raise  # Reraise HTTPException for specific error handling
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/voice_effect")
async def upload_audio(
        audio_file: UploadFile = File(...),
        effect_name: str = Form(...),
        effect_start: int = Form(...), effect_strength: int = Form(3)):
    directory = "./effects_sounds"
    available_effects = [os.path.splitext(file)[0] for file in os.listdir(directory) if
                         os.path.isfile(os.path.join(directory, file))]
    if f'{effect_name}.wav' not in os.listdir(directory):
        raise HTTPException(status_code=400, detail=f"Invalid effect. Available effects are: {available_effects}")

    try:
        # Read the uploaded audio into memory
        audio_bytes = await audio_file.read()
        temp_file_path = "audio_file1.wav"

        # Open the file in binary write mode
        with open(temp_file_path, "wb") as audio_file:
            audio_file.write(audio_bytes)

        audio_data, sr = load_audio(temp_file_path)

        # Apply the chosen effect
        # effect_function = voice_effect_functions[effect]
        processed_audio = apply_effect(audio_data, sr, effect_name, start_effect=effect_start, factor=effect_strength)
        # Convert the processed audio to bytes

        output_bytes = io.BytesIO()
        sf.write(output_bytes, processed_audio, sr, format='wav')
        output_bytes.seek(0)
        # Return the processed audio as a streaming response
        return StreamingResponse(output_bytes, media_type="audio/wav")
    except HTTPException:
        raise  # Reraise HTTPException for specific error handling
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
