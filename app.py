from flask import Flask, jsonify, request, render_template_string
from oracle_clip import OracleClip
from translator import Translator

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip
MODEL = None

@app.route("/helloworld")
def hello():
    return "Hello World!"

@app.route('/query')
def search_for_sentence():
    sentence = request.args.get('sentence', '')
    res = MODEL.find_similar_games(sentence)
    return jsonify(res)

@app.route('/search')
def search():
    query = request.args.get("q", "")

    query_en = TRANSLATOR.translate(query)
    print(f"Query: '{query}' -> '{query_en}'")

    res = MODEL.find_similar_games(query_en) if query_en else []

    with open("static/search.html", 'r', encoding='utf-8') as f:
        html_template = f.read()

    return render_template_string(html_template, query=query, results=res)


if __name__ == '__main__':
    MODEL = OracleClip("data/embeddings/clip-db.pkl", "data/cdrom-db.csv")
    TRANSLATOR = Translator()

    from waitress import serve
    import logging

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    serve(app, host='0.0.0.0', port=7823)
