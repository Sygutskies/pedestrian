import os

directory = "pedestrian/is_project/only_kth/walking"
for i, filename in enumerate(os.listdir(directory)):
    command = "ffmpeg -i " + directory + "/" + str(filename) + " -filter:v fps=fps=25 " + directory + "/" + "stand_" + str(i+1) + ".mp4"
    print(command)
    os.system(command)