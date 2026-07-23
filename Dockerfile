FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg wget unzip libvulkan1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir runpod
RUN mkdir -p /opt && cd /opt \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && unzip -q realesrgan-ncnn-vulkan-20220424-ubuntu.zip -d realesrgan \
    && rm -f realesrgan-ncnn-vulkan-20220424-ubuntu.zip \
    && chmod +x realesrgan/realesrgan-ncnn-vulkan
COPY handler.py /handler.py
CMD ["python3","-u","/handler.py"]
