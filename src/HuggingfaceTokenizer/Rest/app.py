from flask import Flask, request, jsonify
from transformers import AutoTokenizer

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained('/tokenizer/')

@app.route('/tokenize', methods=['POST'])
def tokenize_text():
    text = request.json.get('text')
    batch_enc = tokenizer(text, padding=True, truncation=True)
    return jsonify(batch_enc.data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
