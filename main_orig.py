
import numpy as np
import pandas as pd
import scipy.sparse as sp
import ast
import json

import jax
import jax.numpy as jnp
from jax import Array

from nltk.stem import WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import Bunch

#import matplotlib.pyplot as plt
#import seaborn as sns

from cartm import ContextTopicModel, AttentiveTopicModel
from cartm.preprocessing import (
    CorpusLoader,
    BatchedCorpusLoader,
    build_bow,
)
from cartm.metrics import (
    PerplexityMetric,
    NPMICoherenceMetric,
    SparsityMetric,
    TopicVarianceMetric,
)
from cartm.regularization import DecorrelationRegularization

#sns.set_theme()

from jax import config
config.update("jax_disable_jit", True)

categories = [ 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
]

data = fetch_20newsgroups(data_home='./data/', subset='all').data

#data = fetch_20newsgroups(data_home='./data/', subset='train', categories=categories).data
#data = data[:1000]

filter_mode = 'all'
if filter_mode == 'filtered':
    preprocessor = CorpusLoader(min_token_len=3, max_token_len=20, min_df=5, max_df=0.5, stopwords=set())
if filter_mode == 'all':
    preprocessor = CorpusLoader(min_token_len=1, max_token_len=100, min_df=1, max_df=1.0)
tokenized_data, document_bounds = preprocessor.fit_transform(data)
print(f'Total number of tokens in preprocessed corpus: {len(document_bounds)}')

loader = BatchedCorpusLoader(
    data=tokenized_data,
    doc_bounds=document_bounds,
    batch_size=10000,
    #batch_size=100,
)
print(f'Number of batches: {len(loader)}')

vocab_size = len(preprocessor.vocabulary)
print(f'vocab_size: {vocab_size}')

with open(filter_mode + "_phi_hist_vocab.json", "w") as f:
    json.dump(preprocessor.vocabulary, f, indent=4) 

#bow = build_bow(tokenized_data, document_bounds, vocab_size)

#td = TopicVarianceMetric(top_k=25, tag="TD@25")
perplexity = PerplexityMetric()

decorr = DecorrelationRegularization(0.0, "wt")

model = AttentiveTopicModel(
    vocab_size=len(preprocessor.vocabulary),
    ctx_len=100,
    n_topics=100,
    gamma=0.01,
    metrics=[perplexity],
    regularizers=[decorr],
)

model.fit(
    loader,
    max_iter=3,
    verbose=2,
    seed=42,
    num_batches_before_update=1,
    metric_ratio = 0.05
)

np.save(filter_mode + "_phi_hist.npy", model.phi_hist)

with open(filter_mode + "_phi_hist_perplexity.txt", "w") as f:
    f.write(str(perplexity.history))


