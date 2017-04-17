import locale
import glob
import os.path
import requests
import tarfile
import random
import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import multiprocessing
import numpy as np
from random import sample
from random import shuffle
import datetime
# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 

import os

proxy = 'cvlab:cvlab@http://proxy.miet.ru:3128'

os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cbow")
    parser.add_argument("-size")
    parser.add_argument("-window")
    parser.add_argument("-negative")
    parser.add_argument("-sample")
    parser.add_argument("-iter")
    parser.add_argument("-alpha")
    parser.add_argument("-output")
    parser.add_argument("-threads")
    parser.add_argument("-min_count")
    args = parser.parse_args()
    
    dm = int(args.cbow)
    size = int(args.size)
    window = int(args.window)
    negative = int(args.negative)
    sample = float(args.sample)
    alpha = float(args.alpha)
    passes = int(args.iter)
    cores = int(args.threads)
    min_count = int(args.min_count)
    output = args.output

    SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

    from sklearn.datasets import fetch_20newsgroups
    def get_data(subset):
        newsgroups_data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), download_if_missing=False)
        docs = []
        for news_no, news in enumerate(newsgroups_data.data):    
            tokens = gensim.utils.to_unicode(normalize_text(news)).split()
            
            if len(tokens) == 0:
                continue
            
            split = subset
            sentiment =  newsgroups_data.target[news_no]
            tags = ['SENT_'+ str(news_no) + " " + str(sentiment)]

            docs.append(SentimentDocument(tokens, tags, split, sentiment))
        return docs
    

    train_docs = get_data('train')
    test_docs = get_data('test')
    

    alldocs = train_docs + test_docs

    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))

    assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

    
    model = Doc2Vec(dm=dm, size=size, window=window, alpha = alpha, negative=negative, sample=sample, min_count = min_count, workers=cores, iter=passes)
    

    model.build_vocab(alldocs)

    print("START %s" % datetime.datetime.now())

    
    whole_duration = 0

    duration = 'na'
    with elapsed_timer() as elapsed:
        model.train(train_docs, total_examples = len(train_docs), epochs = model.iter)
        duration = '%.1f' % elapsed()
        whole_duration += elapsed() 

        model.train_words = False
        model.train_labels = True
        model.train(test_docs, total_examples = len(test_docs), epochs = model.iter)

    model.save(output)
    print("END %s" % str(datetime.datetime.now()))
    print("duration %s" % str(whole_duration))
