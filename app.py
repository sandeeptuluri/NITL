from keybert import KeyBERT
from trafilatura import fetch_url, extract
from textblob import TextBlob


from flask import Flask, request, render_template
application = Flask(__name__)

@application.route('/')
def ks():
    return render_template("ks.html")

@application.route('/form_get', methods=['POST','GET'])
def sentiment():
    url = request.form['link']
    html = fetch_url(url)
    data = extract(html)
    data_clean = data.replace("\n"," ").replace("\'", "")
    text = data_clean

    kb = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kb.extract_keywords(text, stop_words='english')

    analysis = TextBlob(text)
    a = analysis.sentiment.polarity
    def type():
        if (a>0):
            return("Positive")
        else:
            return("Negative")
    c = type()

    dc = {}
    dc['KEYWORDS']=keywords
    dc['SENTIMENT']=c
    
    return render_template("ks.html", info=dc)
if __name__ == '__main__':
    application.run(debug=True)
