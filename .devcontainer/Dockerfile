FROM ghcr.io/gdesimone97/cogrob:latest

# Open-vocabulary detection (YOLOE): text encoder used by set_classes()
# ultralytics 8.3.220 + torch/torchvision are already in the base image
USER root
RUN python3 -m pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git" && \
    python3 -m pip install --no-cache-dir numpy==1.26.3
USER mivia

CMD ["/bin/bash"]