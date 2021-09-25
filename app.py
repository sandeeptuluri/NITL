from keybert import KeyBERT
from trafilatura import fetch_url, extract
from textblob import TextBlob
from transformers import pipeline, BartForConditionalGeneration, AutoTokenizer


from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def ks():
    return render_template("ks.html")

@app.route('/form_get', methods=['POST','GET'])
def sentiment():
    url = request.form['link']
    html = fetch_url(url)
    data = extract(html)

    kb = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kb.extract_keywords(data, stop_words='english')

    analysis = TextBlob(data)
    a = analysis.sentiment.polarity
    def type():
        if (a>0):
            return("Positive")
        else:
            return("Negative")
    c = type()


    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    token = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    
    process = pipeline('summarization',model=model, tokenizer=token)
    summary = process(data, truncation=True)

    dc = {}
    dc['KEYWORDS']=keywords
    dc['SENTIMENT']=c
    dc['Summary']=summary
    
    return render_template("ks.html", info=dc)
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=80)
