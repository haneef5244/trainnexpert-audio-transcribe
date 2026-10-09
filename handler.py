import runpod
import os
import urllib.request
from transcriber import transcribe_audio
from chapter_marker import generate_vtt_chapters


def upload_vtt(upload_url: str, vtt_content: str) -> None:
    # The URL is presigned by the API for a PUT with Content-Type text/vtt, so
    # the header must match exactly or S3 rejects the signature.
    request = urllib.request.Request(
        upload_url,
        data=vtt_content.encode("utf-8"),
        method="PUT",
        headers={"Content-Type": "text/vtt"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status not in (200, 204):
            raise RuntimeError(f"VTT upload failed with HTTP {response.status}")

def handler(event):
#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#       
#    Returns:
#       Any: The result to be returned to the client
    try:

        # Extract input data
        print(f"Worker Start")
        input = event['input']
        
        audio_url = input.get('audio_url')
        aws_access_key = input.get('AWS_ACCESS_KEY')
        aws_secret_key = input.get('AWS_SECRET_KEY')
        vtt_upload_url = input.get('vtt_upload_url')
        vtt_s3_key = input.get('vtt_s3_key')

        # Chapters-only: the API sends the course's existing captions as
        # "[<seconds>] text" lines, so there's nothing to transcribe and the
        # captions are left as they are.
        transcript_text = input.get('transcript_text')
        if transcript_text:
            duration = int(input.get('duration_seconds') or 0)
            if duration <= 0:
                raise ValueError("duration_seconds is required with transcript_text")
            chapters = generate_vtt_chapters(transcript_text, duration, aws_access_key, aws_secret_key)
            return {
                "mode": "chapters_only",
                "chapters": chapters.model_dump(),
            }
        
        vtt_to_s3, llm_vtt, duration = transcribe_audio(audio_url)
        chapters = generate_vtt_chapters(llm_vtt, duration, aws_access_key, aws_secret_key)

        # Upload only once chapters succeeded, so a failed job leaves no stray
        # file. Returning just the key keeps the job result small however long
        # the video is; older API versions that send no upload URL still get
        # the VTT inline.
        if vtt_upload_url and vtt_s3_key:
            upload_vtt(vtt_upload_url, vtt_to_s3)
            return {
                "vtt_s3_key": vtt_s3_key,
                "chapters": chapters.model_dump(),
            }

        return {
            "vtt_to_s3": vtt_to_s3,
            "chapters": chapters.model_dump(),
        }
    except Exception as err: 
        raise(err)
     

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })