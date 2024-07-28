script_dir="$(dirname "$(readlink -f "$0")")"

docker build -t pedestrianapp $script_dir