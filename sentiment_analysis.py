import torch
from transformers import pipeline
import pandas as pd

def sentiment_analysis(news):
    def helper(title):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        device = 0 if torch.backends.mps.is_available() else -1
        try:
            title = title[:512]
            sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, device=device)
            analysis = sentiment_pipeline(title)
            if analysis[0]['label'] == 'NEGATIVE':
                return (-1 * analysis[0]['score'])
            else:
                return analysis[0]['score']
        except:
            print(title)
            return 0

    res = []

    for headline in news:
        if type(headline) == str:
            res.append(helper(headline))
        else:
            res.append(0)

    return res

df = pd.read_csv('apple_news.csv')[['Date', 'News']]

analysis = pd.Series(sentiment_analysis(df['News']))

analysis.to_csv("Scores.csv")