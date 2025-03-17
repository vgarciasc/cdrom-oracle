import os
import requests
import shutil
import pandas as pd
import numpy as np
from oracle_clip import OracleClip
from PIL import Image, ImageChops
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

def remove_non_alphanumerics(s):
    return ''.join(e for e in s if e.isalnum() or e == " ")

def get_database_as_dataframe():
    url = "https://docs.google.com/spreadsheets/d/18BAtQoLWMxEYZ-Qwiu6lVp02aGKxAG2u0VHMX7MrXh0/export?format=csv&gid=1036746238"
    response = requests.get(url)
    with open("data/cdrom-db.csv", 'wb') as f:
        f.write(response.content)
    df = pd.DataFrame(pd.read_csv("data/cdrom-db.csv"))
    df['YouTube ID'] = df['Link para gameplay'].apply(lambda x: str(x).split("v=")[-1].split("&t=")[0])
    df['Filename base'] = df['YouTube ID']
    df.to_csv("data/cdrom-db.csv", index=False)
    return df

class VideoProcessor():
    def __init__(self):
        self.df = get_database_as_dataframe()

    def get_youtube_code(self, video_path):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        return filename.split("[")[-1].split("]")[0]

    def get_video_metadata(self, video_path):
        yt_id = self.get_youtube_code(video_path)
        df_row = self.df[self.df['YouTube ID'] == yt_id]
        if df_row.empty:
            return None
        else:
            return df_row.iloc[0].dropna()

    def get_video_filename_base(self, video_path):
        metadata = self.get_video_metadata(video_path)
        if metadata is None:
            return self.get_youtube_code(video_path)
        else:
            return metadata['Filename base']


    def infer_letterbox_crop_bbox(self, video_path):
        with VideoFileClip(video_path) as video:
            duration = int(video.duration)

            # get 5 evenly spaced points
            points = np.arange(0, duration, duration // 5)

            bboxes = []
            for second in points:
                frame_array = video.get_frame(second)
                frame = Image.fromarray(frame_array)

                bg = Image.new(frame.mode, frame.size, frame.getpixel((0, 0)))
                diff = ImageChops.difference(frame, bg)
                diff = ImageChops.add(diff, diff, 2.0, -100)
                bbox = diff.getbbox()
                bboxes.append(bbox)

            # get the most common bbox
            bbox = max(set(bboxes), key=bboxes.count)
            if bbox[0] <= 10 and bbox[1] <= 10:
                return None
            return bbox

    def extract_frames(self, video_path, output_folder, show_progress=True):
        assert os.path.exists(video_path)

        os.makedirs(output_folder, exist_ok=True)

        letterbox_bbox = self.infer_letterbox_crop_bbox(video_path)

        with VideoFileClip(video_path) as video:
            duration = int(video.duration)

            for second in tqdm(range(duration), total=duration, desc="Extracting frames", disable=not show_progress):
                frame_array = video.get_frame(second)
                frame = Image.fromarray(frame_array)
                if letterbox_bbox is not None:
                    frame = frame.crop(letterbox_bbox)
                frame_filename = os.path.join(output_folder, f"{self.get_video_filename_base(video_path)}_{second:04d}.jpg")
                frame.save(frame_filename)

    def eliminate_duplicate_images(self, input_folder, output_folder, threshold=500):
        os.makedirs(output_folder, exist_ok=True)

        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])
        image_array = []
        for image_filename in tqdm(image_files, total=len(image_files), desc="Reading images..."):
            resizer = lambda f: Image.open(f).resize((20, 15)).convert("L")
            image_reduced_np = np.asarray(resizer(f"{input_folder}/{image_filename}"), dtype=np.int8)
            image_array.append((image_filename, image_reduced_np))

        to_copy = np.ones(len(image_array), dtype=bool)
        for i, (file_1, image_np_1) in tqdm(enumerate(image_array), total=len(image_array), desc="Calculating duplicates..."):
            if not to_copy[i]:
                continue

            for j, (file_2, image_np_2) in enumerate(image_array[i + 1:], start=i + 1):
                if not to_copy[j]:
                    continue

                distance = np.linalg.norm(image_np_1 - image_np_2)
                if distance < threshold:
                    to_copy[j] = False

        print(f"{np.sum(to_copy)} unique files remain, out of {len(to_copy)} images.")

        for i, copy in tqdm(enumerate(to_copy), total=len(to_copy), desc="Copying images..."):
            if copy:
                shutil.copy(f"{input_folder}/{image_array[i][0]}", f"{output_folder}/{image_array[i][0]}")

    def delete_saved_frames_from_video(self, video_path, output_folder):
        filename_base = self.get_video_filename_base(video_path)
        if not os.path.exists(output_folder):
            return
        files = os.listdir(output_folder)
        for file in files:
            if file.startswith(filename_base):
                os.remove(os.path.join(output_folder, file))
        print(f"Deleted all {len(files)} saved frames from video '{filename_base}'.")


if __name__ == "__main__":
    vp = VideoProcessor()

    video_path = "../videos/A Casa da Família Urso (1996) - CD-ROM PT-BR [6c8f9m9TfL0].mp4"
    output_folder = f"../frames/{vp.get_video_filename_base(video_path)}"

    vp.delete_saved_frames_from_video(video_path, output_folder)
    vp.extract_frames(video_path, "../_tmp")
    vp.eliminate_duplicate_images("../_tmp", output_folder)
    shutil.rmtree("../_tmp")