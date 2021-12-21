import nltk
import string
from string import punctuation
from newspaper import Article
import tkinter as tk
import transformers
from transformers import pipeline


def summarize():
    url = utext.get('1.0', "end").strip()

    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    data = article.text
    nopunc = [word for word in data if word not in punctuation]
    nopunc = "".join(nopunc)
    summary.config(state='normal')
    process = pipeline('summarization')
    Summary = process(nopunc)

    summary.delete('1.0', 'end')
    summary.insert('1.0', Summary)

    summary.config(state='disabled')


root = tk.Tk()
root.title('Text Summarizer')
root.geometry('1400x800')

ulabel = tk.Label(root, text='Url')
ulabel.pack()

utext = tk.Text(root, height=2, width=150)
utext.pack()

btn = tk.Button(root, text='Summarize', command=summarize)
btn.pack()

slabel = tk.Label(root, text='Summary')
slabel.pack()

summary = tk.Text(root, height=20, width=150)
summary.config(state='disabled', bg='#dddddd')
summary.pack()

root.mainloop()
