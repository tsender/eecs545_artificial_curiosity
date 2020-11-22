docker run --gpus all -it --rm -e DISPLAY=${DISPLAY} -u $(id -u):$(id -g) --name $1 \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw -v $HOME/.Xauthority:$HOME/.Xauthority \
-v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/sudoers.d:/etc/sudoers.d:ro \
-v /etc/shadow:/etc/shadow:ro --network=host \
-v $HOME/umich/eecs545_artificual_curiosity/$1:/$1 \
-w /$1 tsender/tensorflow:latest-gpu