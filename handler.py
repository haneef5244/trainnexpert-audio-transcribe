import runpod
import os
from transcriber import transcribe_audio
from chapter_marker import generate_vtt_chapters

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
        
        vtt_to_s3, llm_vtt, duration = transcribe_audio(audio_url)
        chapters = generate_vtt_chapters(llm_vtt, duration, aws_access_key, aws_secret_key)

        return {
            "vtt_to_s3": vtt_to_s3,
            "chapters": chapters.model_dump(),
        }
    except Exception as err: 
        raise(err)
     

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })