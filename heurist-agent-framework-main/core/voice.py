import hashlib
import logging
import os
import uuid
from pathlib import Path
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Load .env file before using os.getenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch API Key after loading dotenv
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("Missing OpenAI API Key. Ensure it is set in .env or system variables.")
    raise ValueError("Missing OpenAI API Key. Set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes an audio file using OpenAI's Whisper model.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        str: Transcribed text from the audio.

    Raises:
        OpenAIError: If transcription fails.
    """
    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        logger.info(f"Transcription successful for: {file_path}")
        return response.text

    except OpenAIError as e:
        logger.error(f"Failed to transcribe audio ({file_path}): {e}")
        return "Transcription failed."

def speak_text(text: str) -> str:
    """
    Converts text to speech using OpenAI's TTS model and saves it as an MP3 file.

    Args:
        text (str): The input text to convert to speech.

    Returns:
        str: The path to the saved MP3 file.

    Raises:
        OpenAIError: If TTS generation fails.
    """
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )

        # Define the directory for saving audio files
        project_root = Path(__file__).parent.parent
        audio_dir = project_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename
        filename = hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:8] + ".mp3"
        file_path = audio_dir / filename

        # Save the generated audio file
        response.stream_to_file(file_path)

        logger.info(f"TTS audio saved: {file_path}")
        return str(file_path)

    except OpenAIError as e:
        logger.error(f"Failed to generate speech: {e}")
        return "TTS generation failed."
