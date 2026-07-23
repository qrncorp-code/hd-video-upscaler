FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg wget libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir "numpy<2" opencv-python-headless runpod \
      realesrgan==0.3.0 gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0
RUN F=$(python3 -c "import basicsr,os;print(os.path.join(os.path.dirname(basicsr.__file__),'data','degradations.py'))") && \
    sed -i 's/functional_tensor/functional/g' "$F" || true
RUN mkdir -p /models && \
    wget -q -O /models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && \
    wget -q -O /models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
RUN FXW=$(python3 -c "import facexlib,os;d=os.path.join(os.path.dirname(facexlib.__file__),'weights');os.makedirs(d,exist_ok=True);print(d)") && \
    wget -q -O "$FXW/detection_Resnet50_Final.pth" https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O "$FXW/parsing_parsenet.pth" https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
COPY handler.py /handler.py
CMD ["python3","-u","/handler.py"]
