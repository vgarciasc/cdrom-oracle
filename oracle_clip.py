import os
import torch
import open_clip
import pickle
import PIL.Image

class OracleClip():
    def __init__(self, image_embeddings, image_paths, model_id='hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K', device="cpu"):
        print("Loading model...")
        self.device = device
        self.image_embeddings = image_embeddings
        self.image_paths = image_paths
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_id, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_id)
        print("Loaded.")

    def find_most_similar_images(self, sentence, k=3):
        """Find the image most similar to the given sentence."""
        with torch.no_grad():
            text_embedding = self.model.encode_text(open_clip.tokenize([sentence]))
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        similarities = (self.image_embeddings @ text_embedding.T).squeeze(1)
        most_similar_indices = similarities.argsort(descending=True)[:k]

        return [(self.image_paths[i], similarities[i].item()) for i in most_similar_indices]

    def embed_images(self, image_folder):
        """Embed all images in a folder using CLIP."""

        from tqdm import tqdm

        image_embeddings = []
        image_paths = []
        file_paths = os.listdir(image_folder)

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

        return torch.cat(image_embeddings), image_paths

    def update_embeddings(self, image_folder):
        self.image_embeddings, self.image_paths = self.embed_images(image_folder)

    def save_model(self, path):
        data = self.image_embeddings, self.image_paths
        with open(path, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    image_embeddings, image_paths = pickle.load(open("embeddings/image-embeddings-v0.pkl", "rb"))
    image_paths = [path.replace("\\", "/") for path in image_paths]

    model = OracleClip(image_embeddings, image_paths)
    res = model.find_most_similar_images("a cat", k=3)

    for i, (image_path, similarity) in enumerate(res):
        print(f"RANK #{i + 1}:")
        print(f"\tIMAGE: {image_path}")
        print(f"\tSIMILARITY SCORE: {similarity * 100:.2f}%")
        print("-" * 20)
