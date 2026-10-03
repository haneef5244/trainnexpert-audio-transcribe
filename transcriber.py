import os
import uuid
import urllib.request

from faster_whisper import WhisperModel
import math


# Loaded once at module level, not per call — avoids reloading the models
# (and re-downloading from Hugging Face) on every transcription run.
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

MAX_CUE_DURATION = 6.0   # seconds
MAX_CUE_CHARS = 80
MAX_GAP_SECONDS = 1.0    # pause long enough to justify a new cue


def convert_seconds_to_vtt_timestamp(seconds: float):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"


def transcribe_audio(audio_url: str):
    unique_id = uuid.uuid4().hex
    local_audio_path = f"local_audio_{unique_id}.mp3"

    try:
        urllib.request.urlretrieve(audio_url, local_audio_path)

        segments, info = whisper_model.transcribe(
            local_audio_path,
            language=None,
            vad_filter=True,
            word_timestamps=True,
        )
        words = [w for segment in segments for w in segment.words]

        # --- Merge words into cues: break on long pause, max duration, or max length ---
        cues = []
        current_words = []

        for word in words:
            if current_words:
                gap = word.start - current_words[-1].end
                text_so_far = "".join(w.word for w in current_words)
                duration_so_far = current_words[-1].end - current_words[0].start
                if gap > MAX_GAP_SECONDS or duration_so_far > MAX_CUE_DURATION or len(text_so_far) > MAX_CUE_CHARS:
                    cues.append(current_words)
                    current_words = []
            current_words.append(word)

        if current_words:
            cues.append(current_words)

        # --- Each cue's start/end/text, derived once, used for both outputs ---
        vtt_text = "WEBVTT"
        llm_vtt_input = ""
        for i, cue_words in enumerate(cues):
            start = cue_words[0].start
            end = cue_words[-1].end
            text = "".join(w.word for w in cue_words).strip()

            vtt_text += f"\n\n{i+1}\n{convert_seconds_to_vtt_timestamp(start)} --> {convert_seconds_to_vtt_timestamp(end)}\n{text}"
            llm_vtt_input += f"[{int(start)}] {text}\n"

        return vtt_text.strip(), llm_vtt_input.strip(), int(info.duration)

    finally:
        if os.path.exists(local_audio_path):
            os.remove(local_audio_path)
        print("Done!")


