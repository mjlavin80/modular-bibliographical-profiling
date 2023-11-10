
from nltk.corpus import stopwords
from nltk import ngrams, word_tokenize
from fuzzysearch import find_near_matches
import re
from Levenshtein import distance
from collections import Counter 
import numpy as np

# make list of titles 
titles = """Doctor,Dr,Mr,Mrs,Miss,Msgr,Monsignor,Rev,Reverend,Hon,Honorable,Honourable,Prof,Professor,Madame,Madam,Lady,Lord,Sir,Dame,Master,Mistress,Princess,Prince,Duke,Duchess,Baron,Father,Chancellor,Principal,President,Pres,Warden,Dean,Regent,Rector,Provost,Director"""
title_list = titles.rstrip().split(',')

pub_ends = ['company','co','incorporated','inc','firm','press','group','publishers','publishing', \
            'corp', 'publications','pub','books','ltd','limited','society','house','associates', \
            'assoc', 'book', 'university', 'univ', 'brothers', 'bros']

def make_ngram_store(review_store, n, lower=False):
    ngram_store = {}
    for i,j in review_store.items():
        if lower:
            r = [a.lower() for a in j]
        else:
            r = j
            
        these_ngrams = [z for z in ngrams(r, n)]
        ngram_store[i] = Counter(these_ngrams)
    return ngram_store

#ngram_store = make_ngram_store(review_store, 1, lower=True)
#ngram_store[136477222][('library', 'table')]

def preprocess_text(text):
    text = " ".join(re.sub(r"(;|:|\"|&|'(?!s)|,|-|!)", r' \1 ', text).split()) # second semi-colon should be original match text
    tokens = word_tokenize(text)
    processed_text = [''.join([z for z in i if z.isalpha()]) for i in tokens]
    processed_text = [i for i in processed_text if len(i) > 0]
    return processed_text

def drop_tail(name_list):
    if len(name_list) < 1:
        return name_list
    if name_list[-1][0].isupper():
        return name_list
    else:
        name_list = name_list[0:-1]
        return drop_tail(name_list)
    
def get_caps_after(trimmed, title_list):
    this_name = []
    count = 0
    for t in trimmed:
        if count > 2 or t in title_list:
            break  
        if t[0].isupper() and t.lower() not in stopwords.words('english'):
            this_name.append(t)
        else:
            if t.lower() not in stopwords.words('english'):
                this_name.append(t)
                count +=1
            else:
                break
    return drop_tail(this_name)
  
def make_author_candidates(text, title_list):    
    these_names = []
    for e, i in enumerate(text):
        if i in title_list:
            trimmed = text[e+1:]
            this_name = get_caps_after(trimmed, title_list)
            if len(this_name) > 0 and this_name not in these_names:
                these_names.append(this_name)
    return these_names

def associated_names(text, name_seeds, title_list, fuzzy=False, level=0):
    surnames = [i[-1] for i in name_seeds]
    associated = []
    if fuzzy:
        for can in surnames:
            if len(can) > 0:
                for u in text:
                    if len(u) > 0:
                        m = find_near_matches(can, u, max_l_dist=level)
                        for z in m:
                            if z.matched not in surnames:
                                surnames.append(z.matched)
    for e, i in enumerate(text):
        for n in surnames:
            if i == n:
                tail = text[0:e+1]
                tail.reverse()
                this_associated = get_caps_after(tail, title_list)
                this_associated.reverse()
                if this_associated not in associated:
                    associated.append(this_associated)     
    return associated

def drop_short_tail(tokens):
    if len(tokens) > 0:
        if len(tokens[-1]) < 3:
            tokens = tokens[:-1]
            return drop_short_tail(tokens)
    return tokens

def derive_surnames(row, target_column):
    tokens = preprocess_text(str(row[target_column])) 
    tokens = drop_short_tail(tokens)
    if len(tokens) > 0:
        return [tokens[-1].lower(),]
    else:
        return []

def find_ngrams(row, id_col, target_col, review_store, n_col, max_len):
    """ match token sequence (len 1,2,3 ... n, etc.) to set of review tokens represented as ngrams of same length"""
    
    this_review = [i.lower() for i in review_store[row[id_col]]]
    n = row[n_col]
    if n==0:
        return 0
    if n > 1:
        if len(row[target_col]) > max_len:
            match_grams = tuple(row[target_col][0:max_len])
            n = max_len
        else:
            match_grams = tuple(row[target_col])
        these_ngrams = [i for i in ngrams(this_review, n)]

        if match_grams in these_ngrams:
            return 1
    # if column value is list with len of 1
    if n == 1:
        if row[target_col][0] in this_review:
            return 1
    return 0

def find_fuzzy(row, id_col, target_col, review_store, fuzz, lower=False):
    """ fuzzy match string (or token ngrams converted to string) in review text """
    
    if lower:
        this_review = review_store[row[id_col]].lower()
    else: 
        this_review = review_store[row[id_col]]
    
    if type(row[target_col]) == list:
        match_txt = ' '.join(row[target_col])
    else:
        match_txt = row[target_col]
    if match_txt != '':
        m = find_near_matches(match_txt, this_review, max_l_dist=fuzz)
        output = [z for z in m]
        if len(output) > 0:
            return 1
    return 0

def find_ngrams_fp(row, id_col, target_col, ngram_store, n_col, max_len):
    """ match token sequence (len 1,2,3 ... n, etc.) 
    to all sets of review token ngrams and return fp rate"""
    fp_count = 0
    matching_ids = []
    n = row[n_col]
    if n > 0:
        # limit length of n-gram if too long    
        if len(row[target_col]) > max_len:
            n = max_len
            match_grams = tuple(row[target_col][0:max_len])
        else:
            match_grams = tuple(row[target_col])
        # set counters source based on value of N
        review_counters = ngram_store[n]
        
        # each r is a review ID
        for r in review_counters.keys():
            # check if match grams in counter
            
            if r != row[id_col]:
                if r not in row['record_id_matches']:
                    if review_counters[r][match_grams] > 0:
                        fp_count += 1
                        matching_ids.append(r)
          
    return fp_count, matching_ids

def find_fuzzy_fp(row, id_col, target_col, ngram_store, n_col, max_len, fuzz, lower=False):
    """ fuzzy match string (or token ngrams converted to string) in review text """
    fp_count = 0 
    n = row[n_col]
    if n > 0:    
        if len(row[target_col]) > max_len:
            n = max_len
            match_txt = ' '.join(row[target_col][0:max_len])
        else:
            match_txt = ' '.join(row[target_col])
            
        all_reviews = ngram_store[n]
        if match_txt != '':
            for k in all_reviews.keys():
                this_review = all_reviews[k]
                if k != row[id_col]:
                    if k not in row['record_id_matches']:
                        for ngram in this_review.keys():
                            if lower:
                                match_txt = match_txt.lower()
                                ngram = [o.lower() for o in ngram]
                            match_gram = ' '.join(ngram)
                            d = distance(match_txt, match_gram)

                            if d <= fuzz:
                                fp_count += 1
                                break
    return fp_count

def make_lev_distance_matrix(rows, columns):
    """ Construct a levenshtein distance matrix for a list of strings"""
    dist_matrix = np.zeros(shape=(len(columns), len(rows)))
    for i in range(0, len(columns)):
        for j in range(0, len(rows)):
            if columns[i] == rows[j]:
                dist_matrix[i][j] = 0
            else:
                dist_matrix[i][j] = distance(columns[i], rows[j]) 
    return dist_matrix

def cull_list_of_dicts(term_list, list_of_dicts):
    results = []
    for d in list_of_dicts:
        result = {}
        for term in term_list:
            try:
                result[term] = d[term]
            except:
                pass 
        results.append(Counter(result))
    return results

def remove_from_list_of_dicts(term_list, list_of_dicts):
    results = []
    for d in list_of_dicts:
        result = {}
        for term in term_list:
            try:
                del d[term]
            except:
                pass 
        results.append(Counter(d))
    return results