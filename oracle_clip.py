import os
import shutil
from datetime import datetime

import torch
import open_clip
import pickle
import PIL.Image
import pandas as pd
from tqdm import tqdm
import numpy as np

class OracleClip():
    def __init__(self, embedding_db_path, metadata_db_path, embeddings_folder_path="data/embeddings",
                 device="cpu", model_id='hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        print("Initialized OracleClip.")
        print("Loading embeddings db...")
        if embedding_db_path is not None and os.path.exists(embedding_db_path):
            with open(embedding_db_path, "rb") as f:
                self.database = pd.read_pickle(f)
                self.image_embeddings = torch.tensor(self.database['embeddings'].values.tolist())
                print(f"Loaded database with {len(self.database)} images.")
        else:
            self.database = pd.DataFrame(columns=["paths", "embeddings", "youtube_id"])
            self.image_embeddings = torch.tensor(np.zeros((0, 512)))
            print("No database found. Starting with an empty database.")

        print("Loading metadata...")
        self.metadata_database = pd.read_csv(metadata_db_path)
        self.embeddings_folder_path = embeddings_folder_path
        print("Loaded metadata.")

        print("Loading CLIP model...")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        print("Loaded CLIP model.")

    def calculate_similarity_to_frames(self, sentence):
        """Find the image most similar to the given sentence."""
        with torch.no_grad():
            text_embedding = self.model.encode_text(open_clip.tokenize([sentence]))
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        similarities = (self.image_embeddings @ text_embedding.T).squeeze(1)

        df = self.database.copy()
        df["similarity"] = similarities.tolist()
        df.sort_values("similarity", ascending=False, inplace=True)
        return df[df['similarity'] > 0.25][['youtube_id', 'paths', 'similarity']]

    def find_similar_games(self, sentence, max_games=5, max_images_per_game=6):
        """Find the games most similar to the given sentence."""

        id_cols = ['youtube_id', 'YouTube ID', 'Nome da obra no Brasil', 'Lançamento brasileiro']

        df = self.calculate_similarity_to_frames(sentence)
        df = df.merge(self.metadata_database, left_on="youtube_id", right_on="YouTube ID (capitalized)")
        df = df.groupby(id_cols, sort=False).agg({'paths': list, 'similarity': list})
        df['paths'] = df['paths'].apply(lambda x: x[:max_images_per_game])
        df = df.head(max_games)
        res = df.to_dict(orient='index')
        res = [{**{c: x[i] for i, c in enumerate(id_cols)}, **res[x]} for x in res.keys()]
        return res

    def get_folder_embeddings(self, image_folder):
        """Embed all images in a folder using CLIP."""

        file_paths = os.listdir(image_folder)

        image_embeddings = []
        image_paths = []
        for filename in tqdm(file_paths, total=len(file_paths), desc="Embedding images..."):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(image_folder, filename)
                image = self.preprocess(PIL.Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                image_embeddings.append(image_features)
                image_paths.append(image_path)

        if len(image_embeddings) == 0:
            raise ValueError("No valid images found in the folder.")

        image_embeddings = torch.cat(image_embeddings)
        image_paths = [x.split("\\")[-1] for x in image_paths]
        return image_embeddings, image_paths

    def update_database_embeddings(self, new_image_folder):
        os.makedirs(self.embeddings_folder_path, exist_ok=True)

        youtube_id = new_image_folder.split("/")[-1].split("\\")[-1].split(".")[0][:10]

        new_embeddings, new_paths = self.get_folder_embeddings(new_image_folder)
        new_db = pd.DataFrame({"paths": new_paths, "embeddings": new_embeddings.detach().cpu().numpy().tolist()})
        new_db["youtube_id"] = youtube_id

        old_db = self.database[self.database["youtube_id"] != youtube_id]
        db = pd.concat([old_db, new_db], ignore_index=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        db.to_pickle(f"{self.embeddings_folder_path}/clip-db_{timestamp}.pkl")
        shutil.copy(f"{self.embeddings_folder_path}/clip-db_{timestamp}.pkl", f"{self.embeddings_folder_path}/clip-db.pkl")
        self.database = db

        print(f"Updated database with {len(new_paths)} new images.")
        print(f"Current database contains {db['youtube_id'].nunique()} games and {len(db)} images.")

if __name__ == "__main__":
    model = OracleClip("data/embeddings/clip-db.pkl", "data/cdrom-db.csv")
    res = model.find_similar_games("Cartoon drawing of an alien")
    print(res)
