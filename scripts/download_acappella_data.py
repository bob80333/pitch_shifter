import os
import subprocess
import csv

dataset_path = "dataset_dir/acappella_dataset"

if __name__ == "__main__":
    dirs = ["train", "val", "test"]

    for dir in dirs:
        # with open(os.path.join(dataset_path, dir+".csv"), "r", encoding="utf-8") as f:
        #     reader = csv.reader(f)
        #     next(reader) # skip header

        #     # header: ID,Repeat,Init,Fin,Length (s),Song Name,Link,Comments,Singer,Language,Gender
        #     for row in reader:
        #         start = row[2].replace(".", ":")
        #         end = row[3].replace(".", ":")
        #         name = row[0]
        #         link = row[6]
        #         # download the file with yt-dlp

        #         command = f"yt-dlp --quiet --no-warnings --force-overwrites -o {dataset_path}/{dir}_raw/{name}.%(ext)s -f bestaudio {link} --download-sections \"*{start}-{end}\""

        #         out = subprocess.run(command, shell=True)

        
        # convert and split the files:

        # make output directory
        os.makedirs(os.path.join(dataset_path, dir), exist_ok=True)

        # get all files:
        files = os.listdir(os.path.join(dataset_path, dir+"_raw"))
        for file in files:
            # convert to wav and segment into 6s segments
            # get the file name without extension
            name = os.path.splitext(file)[0]
            command = f"ffmpeg -hide_banner -loglevel error -i {dataset_path}/{dir}_raw/{file} -ac 1 -ar 48000 -y -f segment -segment_time 6 -reset_timestamps 1 {dataset_path}/{dir}/{name}_%03d.wav"
            out = subprocess.run(command, shell=True)

