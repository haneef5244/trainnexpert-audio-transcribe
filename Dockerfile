FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install dependencies
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py transcriber.py chapter_marker.py /

# Start the container
CMD ["python3", "-u", "handler.py"]
