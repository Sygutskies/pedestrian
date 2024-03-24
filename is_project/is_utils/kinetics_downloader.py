from pytube import YouTube
import ffmpeg
import json

def seconds_to_hh_mm_ss(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

with open('kinetics700_2020/train.json', 'r') as json_file:
    data = json.load(json_file)

id = 0
for video_id, video_info in data.items():
    if "annotations" in video_info and "label" in video_info["annotations"] and video_info["annotations"]["label"] == "talking on cell phone":
        video_url = video_info["url"]
        yt = YouTube(video_url)
        try:
            stream = yt.streams.get_highest_resolution()
            output_path = 'cut_train'
            filename = "jogging" + str(id) + ".webm"
            start = seconds_to_hh_mm_ss(video_info["annotations"]["segment"][0])
            stream.download(output_path, filename)
            (
                ffmpeg
                .input('cut_train/' + filename, ss=str(start), t='00:00:10')
                .output('cut_train/out_' + filename)
                .run()
            )
            id += 1
        except:
            print("Movie not exists")