def preprocessor(data):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    scaler = StandardScaler()
    temp_labels = data.columns
    scaler.fit(data)
    preprocessed_data=scaler.transform(data)
    preprocessed_data=pd.DataFrame(preprocessed_data)
    preprocessed_data.columns=temp_labels

    return preprocessed_data


def iso_remove(X, y, contamination=0.025):
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(contamination=contamination)
    yhat = iso.fit_predict(X)
    X = X[yhat == 1]
    y = y[yhat == 1]
    return X, y


def local_remove(X, y):
    from sklearn.neighbors import LocalOutlierFactor

    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    X= X[yhat == 1]
    y = y[yhat == 1]
    return X, y


def clean(text):
    import re

    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.lower()
    return text


def token_stop(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    temp = clean(text)
    tokens = word_tokenize(temp)
    newlist = []
    for word in tokens:
        if word not in set(stopwords.words('english')):
            newlist.append(word)
    return newlist


def vader_score(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize
    analyzer = SentimentIntensityAnalyzer()
    result = {'positive': [],
              'neutral': [],
              'negative': [],
              'compound': [],
              'polarity': []}
    for text in texts:
        sentences = sent_tokenize(text)
        pos = compound = neu = neg = 0
        for sentence in sentences:
            vs = analyzer.polarity_scores(sentence)
            pos += vs['pos'] / (len(sentences))
            neu += vs['neu'] / (len(sentences))
            neg += vs['neg'] / (len(sentences))
            compound += vs['compound'] / (len(sentences))
        result['positive'].append(pos)
        result['neutral'].append(neu)
        result['negative'].append(neg)
        result['compound'].append(compound)
        result['polarity'].append(getPolarity(compound))
    return result


def getPolarity(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
