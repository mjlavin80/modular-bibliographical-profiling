def parse_genre(row):
    text = str(row['reviewed_book_genre']).lower().replace('non-fiction', 'nonfiction')
    mylist = text.split('-')
    stripped = [i.strip() for i in mylist]
    if len(stripped) > 1:
        if stripped[1] in ['poems', 'poetry']:
            return 'poetry'
        elif  stripped[1] == 'play':
            return 'drama'
        else:
            return stripped[0]
    else:
        if stripped[0] in ['religion', 'dictionary', 'lectures']:
            return 'nonfiction'
        else:
            return stripped[0]

def split_on_del_w_id(df, id_col, target_col, d):
    records = []
    for e, row in df.iterrows():
        text = str(row[target_col])
        entries = text.split(d)
        for i in entries:
            records.append([row[id_col], i]) 
    return pd.DataFrame.from_records(records, columns=[id_col, target_col])

