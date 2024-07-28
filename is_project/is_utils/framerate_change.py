import os

directory = "final_test/"
for i, filename in enumerate(os.listdir(directory)):
    command = "ffmpeg -i " + directory + "/" + str(filename) + " -filter:v fps=fps=25 " + directory + "/" + "walking" + str(i+1) + ".mp4"
    print(command)
    os.system(command)