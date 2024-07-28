script_dir="$(dirname "$(readlink -f "$0")")"

docker run -v $script_dir/../../:/pedestrian -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it pedestrianapp