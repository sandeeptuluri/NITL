from keybert import KeyBERT
from textblob import TextBlob
from trafilatura import fetch_url, extract
from flask import Flask, request , jsonify
app = Flask(__name__)

import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


@app.route('/')
def ks():
    return render_template("ks.html")

@app.route('/result', methods=['POST','GET'])
def result():
    url = request.form['Link']
    html = fetch_url(url)
    text = extract(html)


    kb = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kb.extract_keywords(text, stop_words='english')

    analysis = TextBlob(text)
    a = analysis.sentiment.polarity
    def type():
        if (a>0):
            return("Positive")
        elif (a<0):
            return("Negative")
        else:
            return("Neutral")
    c = type()

    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base',return_dict=True)
    device = torch.device('cpu')

    inputs = tokenizer.encode("summarize: " + t5_prepared_Text, return_tensors='pt', max_length=512, truncation=True)

    summary_ids = model.generate(inputs,num_beams=2,min_length=80,max_length=150,length_penalty=5.)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    dc={
        'KEYWORDS':keywords,
        'SENTIMENT':c,
        'SUMMARY':summary
        }
    return jsonify(dc)
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=7645,debug=True)
