import pickle

from flask import Flask, jsonify, request
from datetime import datetime
from oracle_clip import OracleClip

app = Flask(__name__)
MODEL = None

@app.route('/query')
def search_for_sentence():
    sentence = request.args.get('sentence', '')
    res = MODEL.find_most_similar_images(sentence, k=3)
    return jsonify(res)

if __name__ == '__main__':
    image_embeddings, image_paths = pickle.load(open("embeddings/image-embeddings-v0.pkl", "rb"))
    image_paths = [path.replace("\\", "/") for path in image_paths]
    MODEL = OracleClip(image_embeddings, image_paths)

    app.run(host='0.0.0.0', port=5000)