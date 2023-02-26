def continent(country):
    country_map = {
        "Argentina": "South America",
        "Armenia": "Europe",
        "Australia": "Oceania",
        "Austria": "Europe",
        "Bosnia and Herzegovina": "Europe",
        "Brazil": "South America",
        "Bulgaria": "Europe",
        "Canada": "North America",
        "Chile": "South America",
        "China": "Asia",
        "Croatia": "Europe",
        "Cyprus": "Europe",
        "Czech Republic": "Europe",
        "Egypt": "Africa",
        "England": "Europe",
        "France": "Europe",
        "Georgia": "North America",
        "Germany": "Europe",
        "Greece": "Europe",
        "Hungary": "Europe",
        "India": "Asia",
        "Israel": "Asia",
        "Italy": "Europe",
        "Lebanon": "Asia",
        "Luxembourg": "Europe",
        "Macedonia": "Europe",
        "Mexico": "North America",
        "Moldova": "Europe",
        "Morocco": "Africa",
        "New Zealand": "Oceania",
        "Peru": "South America",
        "Portugal": "Europe",
        "Romania": "Europe",
        "Serbia": "Europe",
        "Slovakia": "Europe",
        "Slovenia": "Europe",
        "South Africa": "Africa",
        "Spain": "Europe",
        "Switzerland": "Europe",
        "Turkey": "Asia",
        "US": "North America",
        "Ukraine": "Europe",
        "Uruguay": "South America"
    }

    return country.map(country_map)


def process_new_data(df_wine, df_wine_400, lda, dictionary, scaler, model_xgb,
                     price, description, country, province, variety, year):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from Preprocessing import clean, token_stop, vader_score, getPolarity
    import json

    new_data = pd.DataFrame({'price': [price],
                             'description': description,
                             'country': country,
                             'province': province,
                             'variety': variety,
                             'year': year})

    # generate vader score
    vader_result = vader_score(new_data['description'])
    new_data['positive'] = vader_result['positive']
    new_data['neutral'] = vader_result['neutral']
    new_data['negative'] = vader_result['negative']
    new_data['compound'] = vader_result['compound']
    new_data['polarity'] = vader_result['polarity']

    # generate continent
    new_data['continent'] = continent(new_data['country'])

    # generate topic distribution
    new_doc = token_stop(new_data.iloc[0, 1])
    new_doc_bow = dictionary.doc2bow(new_doc)
    new_doc_dist = lda.get_document_topics(new_doc_bow)
    dist = np.zeros(140, )
    for (i, prob) in new_doc_dist:
        dist[i] = prob
    new_doc_dist = dist
    new_data = pd.concat([new_data, pd.DataFrame({str(i): [value] for i, value in enumerate(pd.Series(new_doc_dist))})],
                         axis=1)

    # standardize
    new_data_ready = pd.concat(
        [new_data[['price', 'positive', 'neutral', 'negative', 'compound']], new_data.iloc[:, 12:]], axis=1)
    labels = list(new_data_ready.columns)
    new_data_ready = pd.DataFrame(scaler.transform(new_data_ready))
    new_data_ready.columns = labels

    # one hot encoder
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(new_data[['country', 'continent', 'polarity', 'year', 'variety']])
    df_ohe = pd.DataFrame(transformed.toarray())
    col_name = []
    for i in ohe.categories_:
        col_name.extend(list(i))
    df_ohe.columns = col_name
    new_data_ready = pd.concat([new_data_ready, df_ohe], axis=1)

    # prepare data for xgb model
    new_data_xgb = pd.DataFrame({i: [0] for i in list(df_wine_400.columns)[1:]})
    xgb_columns = list(new_data_xgb.columns)
    for i in xgb_columns:
        try:
            new_data_xgb[i] = new_data_ready[i]
        except:
            continue

    f = open('data/ohe_reference.json')
    ohe_reference = json.load(f)
    f.close()

    for key, value in ohe_reference.items():
        new_data_xgb[key] = value[int(new_data_xgb[key])]

    predicted_new_data = pd.DataFrame({'normalized rating': model_xgb.predict(new_data_xgb)})
    new_data_xgb = pd.concat([predicted_new_data, new_data_xgb], axis=1)

    return new_data_xgb






