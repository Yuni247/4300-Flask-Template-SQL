from collections import defaultdict
from collections import Counter
import json
import math
import string
import time
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from IPython.core.display import HTML

treebank_tokenizer = TreebankWordTokenizer()

def build_inverted_index(msgs):
    """ Builds an inverted index from the messages.
    
    Arguments
    =========
    
    msgs: list of dicts.
        Each message in this list already has a 'toks'
        field that contains the tokenized message.
    
    Returns
    =======
    
    inverted_index: dict
        For each term, the index contains 
        a sorted list of tuples (doc_id, count_of_term_in_doc)
        such that tuples with smaller doc_ids appear first:
        inverted_index[term] = [(d1, tf1), (d2, tf2), ...]
        
    Example
    =======
    
    >> test_idx = build_inverted_index([
    ...    {'toks': ['to', 'be', 'or', 'not', 'to', 'be']},
    ...    {'toks': ['do', 'be', 'do', 'be', 'do']}])
    
    >> test_idx['be']
    [(0, 2), (1, 2)]
    
    >> test_idx['not']
    [(0, 1)]
    
    """
    # YOUR CODE HERE
    d = {}
    # go thru each message of each document
    # structure d as key: word, value: dictionary -> value dictionary has key: docID, value: count
    for msg_index in range(len(msgs)):
        words = msgs[msg_index]['toks']
        for word in words:
            if word not in d:
                d[word] = {msg_index:1}
            else:
                if msg_index in d[word]:
                    d[word][msg_index] += 1
                else:
                    d[word][msg_index] = 1
    for word in d:
        lst = []
        for doc_id in d[word]:
            lst.append((doc_id, d[word][doc_id]))
        d[word] = lst
    return d



def compute_idf(inv_idx, n_docs, min_df=5, max_df_ratio=0.95):
    """ Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.
    
    Hint: Make sure to use log base 2.
    
    Arguments
    =========
    
    inv_idx: an inverted index as above
    
    n_docs: int,
        The number of documents.
        
    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored. 
        Documents that appear min_df number of times should be included.
    
    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.
    
    Returns
    =======
    
    idf: dict
        For each term, the dict contains the idf value.
        
    """
    
    # YOUR CODE HERE
    idf = {}
    for term in inv_idx:
        df = len(inv_idx[term])
        if df >= min_df:
            if df/n_docs <= max_df_ratio:
                frac = n_docs/(1+df)
                idf[term] = math.log(frac, 2)
    return idf

def compute_doc_norms(index, idf, n_docs):
    """ Precompute the euclidean norm of each document.
    
    Arguments
    =========
    
    index: the inverted index as above
    
    idf: dict,
        Precomputed idf values for the terms.
    
    n_docs: int,
        The total number of documents.
    
    Returns
    =======
    
    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """
    
    # YOUR CODE HERE
    norms = np.zeros((n_docs))
    for word in index:
        if word in idf:
            idf_i = idf[word]
            for j in index[word]:
                tf_ij = j[1]
                norms[j[0]] += (tf_ij*idf_i)**2
    norms = norms**(1/2)
    return norms
    
    

def accumulate_dot_scores(query_word_counts, index, idf):
    """ Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    
    Arguments
    =========
    
    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.
    
    index: the inverted index as above,
    
    idf: dict,
        Precomputed idf values for the terms.
    
    Returns
    =======
    
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    
    # YOUR CODE HERE
    doc_scores = {}
    for word in query_word_counts:
        q_i = query_word_counts[word]
        # print(word in index)
        # print(index)
        if word in index:
            ind = index[word]
            for doc in ind:
                # print(doc)
                doc_id = doc[0]
                freq = doc[1]
                if doc_id in doc_scores:
                    doc_scores[doc_id] += idf[word]*q_i*idf[word]*freq
                else:
                    doc_scores[doc_id] = idf[word]*q_i*idf[word]*freq
            
    return doc_scores


def index_search(query, index, idf, doc_norms, scores, rating_dict, thumbs_dict, score_func=accumulate_dot_scores, tokenizer=treebank_tokenizer):
    """ Search the collection of documents for the given query
    
    Arguments
    =========
    
    query: string,
        The query we are looking for.
    
    index: an inverted index as above
    
    idf: idf values precomputed as above
    
    doc_norms: document norms as computed above
    
    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
        (See Q7)
    
    tokenizer: a TreebankWordTokenizer
    
    Returns
    =======
    
    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.
    
    Note: 
        
    """
    
    # YOUR CODE HERE
    def sortFirst(item):
        return item[0]
    lst = []
    query_tokens = tokenizer.tokenize(query)
    query_word_counts = {}
    for word in query_tokens:
        if word in query_word_counts:
            query_word_counts[word] += 1
        else:
            query_word_counts[word] = 1
            
    scores = score_func(query_word_counts, index, idf)
    q_norm = 0
    for word in query_word_counts:
        if word in idf:
            q_norm += (query_word_counts[word]*idf[word])**2
    q_norm = q_norm**(1/2)
    for doc_id in scores:
        denom = q_norm*doc_norms[doc_id]
        sc = (scores[doc_id]/denom)*(1+rating_dict[doc_id]*thumbs_dict[doc_id])
        lst.append((sc, doc_id))
    lst.sort(key=sortFirst, reverse=True)
    return lst

def get_responses_from_results(response, results):
    """ 
    Take results of index search and get list of attractions
    """
    acc = []
    # print(results)
    for x in results:
        id = x[1]
        acc.append(response[id])
    return acc[:21]
