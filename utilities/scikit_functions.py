from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def pairwise_cosine(a, b):
    records = []
    for e, i in enumerate(a):
        for x,y in enumerate(b):
            c = cosine_similarity(i.reshape(1, -1), y.reshape(1, -1))
            records.append([e, x, c[0][0]])          
    return records 

def pairwise_performance(records):
    df = pd.DataFrame.from_records(records, columns=['source', 'target', 'score'])
    df = df.sort_values(by=['source', 'score'], ascending=[True, False])
    
    df_best = df.groupby('source').first().reset_index()
    #accuracy = df_best.loc[df_best['source'] == df_best['target']].shape[0]/df_best.shape[0]
    scores = []
    for i in range(1, len(df)):
        df_top_n = df.groupby('source').head(i)
        score = df_top_n.loc[df_top_n['source'] == df_top_n['target']].shape[0]/df_best.shape[0]
        scores.append([i, score])
        if score == 1.0:
            break
    return scores

def pairwise_df(records):
    df = pd.DataFrame.from_records(records, columns=['source', 'target', 'score'])
    df = df.sort_values(by=['source', 'score'], ascending=[True, False])
    df_best = df.groupby('source').first().reset_index()
    return df_best