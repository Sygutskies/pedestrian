import os

directory = "."
for i, filename in enumerate(os.listdir(directory)):
    command = f'ffmpeg -i {filename} -vf "scale=1920:1080" _{filename}'
    print(command)
    os.system(command)