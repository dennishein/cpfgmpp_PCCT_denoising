docker run -it --gpus 'device=0' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 8889:8889 -v "$(pwd)":/workspace edm:latest
