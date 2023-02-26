def J_S_distance(df_wine, new_data, n, sort_by_points=False):
    import pandas as pd
    import numpy as np
    from scipy.stats import entropy

    def jensen_shannon(query, matrix):
        p = query.T
        q = matrix.T
        m = 0.5 * (p + q)
        return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))

    def get_most_similar_documents(query, matrix, k):
        sims = jensen_shannon(query, matrix)
        return sims.argsort()[:k]

    def print_most_similar(df, query, matrix, k, sort_points=False):
        most_sim_ids = get_most_similar_documents(query=query, matrix=matrix, k=k)
        most_similar_df = df[df.index.isin(most_sim_ids)]
        most_similar_df = most_similar_df[['title', 'normalized rating']]
        if sort_points:
            most_similar_df = most_similar_df.sort_values(by=['normalized rating'], ascending=False)
            print(f'{k} Most similar wines (descending order by similarity and points):')
        else:
            print(f'{k} Most similar wines (descending order by similarity):')
        for i in range(k):
            print(f'{i + 1}. {most_similar_df.iloc[i, 0]} ---- {round(most_similar_df.iloc[i, 1], 2)}')

    doc_topic_dist = np.array(df_wine.iloc[:, 20:])
    print_most_similar(df=df_wine, query=new_data, matrix=doc_topic_dist, k=n, sort_points=sort_by_points)


def cosine_similarity(df_wine, df_wine_400, new_data, n, sort_by_points=False):
    import scipy.spatial as sp
    def print_most_similar(df, query, matrix, k, sort_points=False):
        cos_sims = 1 - sp.distance.cdist(matrix, query, 'cosine')
        cos_sims = cos_sims.reshape(len(cos_sims))
        most_sim_ids = sorted(range(len(cos_sims)), key=lambda i: -cos_sims[i])[:k]
        most_similar_df = df[df.index.isin(most_sim_ids)]
        most_similar_df = most_similar_df[['title', 'normalized rating']]
        if sort_points:
            most_similar_df = most_similar_df.sort_values(by=['normalized rating'], ascending=False)
            print(f'{k} Most similar wines (descending order by similarity and points):')
        else:
            print(f'{k} Most similar wines (descending order by similarity):')
        for i in range(k):
            print(f'{i + 1}. {most_similar_df.iloc[i, 0]} ---- {round(most_similar_df.iloc[i, 1], 2)}')

    print_most_similar(df=df_wine, query=new_data, matrix=df_wine_400, k=n, sort_points=sort_by_points)


def soft_cosine_measure_similarity(df_wine, new_input:str, termsim_matrix, tfidf, corpus, dictionary, k, sort_by_points=False):
    from Preprocessing import token_stop
    import tqdm

    vectors = tfidf[corpus]
    new_vector = tfidf[dictionary.doc2bow(token_stop(new_input))]
    similarity_scores = []
    for i in tqdm.tqdm(range(len(vectors))):
        try:
            similarity = termsim_matrix.inner_product(new_vector, vectors[i], normalized=(True, True))
            similarity_scores.append((i, similarity))
        except:
            continue
    top_k = sorted(similarity_scores, key=lambda x: x[1], reverse = True)[:k]
    top_k_index = [tuple_[0] for tuple_ in top_k]
    title_points = []
    for index in top_k_index:
        title_points.append((df_wine.iloc[index]['title'], round(df_wine.iloc[index]['normalized rating'], 2)))
    if sort_by_points:
        print(f'{k} Most similar wines (descending order by similarity and points):')
        sorted_title_points = sorted(title_points, key=lambda x: x[1], reverse=True)
        for i, tuple_ in enumerate(sorted_title_points):
            print(f'{i+1}. {tuple_[0]} ---- {tuple_[1]}')
    else:
        print(f'{k} Most similar wines (descending order by similarity):')
        for i, tuple_ in enumerate(title_points):
            print(f'{i+1}. {tuple_[0]} ---- {tuple_[1]}')





