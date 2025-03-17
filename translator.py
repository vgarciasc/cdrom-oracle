import json
import deepl

class Translator:
    def __init__(self):
        config = json.loads(open("config.json", "r").read())
        self.deepl_model = deepl.Translator(auth_key=config["deepl_key"])

    def translate(self, text):
        if not text:
            return ""

        if len(text) > 500:
            raise Exception("Query is too long. Maximum length is 500 characters.")

        return self.deepl_model.translate_text(text, source_lang="PT", target_lang="EN-US").text

if __name__ == "__main__":
    translator = Translator()
    print(translator.translate("Menino e menina andando na frente de um castelo"))