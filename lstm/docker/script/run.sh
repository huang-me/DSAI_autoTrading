# !bin/bash

docker run -it --rm --gpus all -v ${PWD}:/home -v /tmp/.X11-unix:/tmp/.X11-unix dsai_hw2 bash