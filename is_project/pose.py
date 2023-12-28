import os
directory1 = 'datasets/pedestrian_alter/test/labels'
directory2 = 'datasets/pedestrian_alter/valid/labels'
directory3 = 'datasets/pedestrian_alter/train/labels'

def change_annotations(directory):
    for filename in os.listdir(directory):
        with open(directory + '/' + filename, "r+") as f:
            new_file = []
            new_line = ''
            for line in f.readlines():
                new_line = '0' + line[1:]
                new_file.append(new_line)
            f.seek(0)
            f.truncate()
            f.writelines(new_file)
            f.close()

change_annotations(directory1)
change_annotations(directory2)
change_annotations(directory3)
