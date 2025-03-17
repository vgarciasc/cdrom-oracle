import glob
import os

from video_processor import VideoProcessor
import shutil
import pickle
import argparse
from tqdm import tqdm
from oracle_clip import OracleClip

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--video_path", type=str)
    args.add_argument("--skip_frame_extraction_if_frame_folder_exists", action="store_true", default=True)
    args.add_argument("--skip_embedding_if_frame_folder_exists", action="store_true")
    args.add_argument("--duplicate_threshold", type=int, default=500)
    args = args.parse_args()

    embedding_db_path = "data/embeddings/clip-db.pkl" if os.path.exists("data/embeddings/clip-db.pkl") else None
    metadata_db_path = "data/cdrom-db.csv" if os.path.exists("data/cdrom-db.csv") else None

    vp = VideoProcessor()
    oc = OracleClip(embedding_db_path, metadata_db_path, device="cuda")

    video_paths = glob.glob(args.video_path) if "*" in args.video_path else [args.video_path]
    for video_path in tqdm(video_paths, total=len(video_paths), desc="Processing videos...", unit="video"):
        filename_base = vp.get_video_filename_base(video_path)
        output_folder = f"frames/{filename_base}"
        print(f"Processing video '{filename_base}'...")

        if os.path.exists(output_folder) and args.skip_embedding_if_frame_folder_exists:
            print(f"Skipping entire '{filename_base}', frames already exist.")
            continue

        if os.path.exists(output_folder) and args.skip_frame_extraction_if_frame_folder_exists:
            print(f"Skipping video extraction of '{filename_base}', frames already exist.")
        else:
            if os.path.exists("_tmp"):
                shutil.rmtree("_tmp")
            vp.delete_saved_frames_from_video(video_path, output_folder)
            vp.extract_frames(video_path, "_tmp")
            vp.eliminate_duplicate_images("_tmp", output_folder, threshold=args.duplicate_threshold)
            shutil.rmtree("_tmp")

        oc.update_database_embeddings(output_folder)
