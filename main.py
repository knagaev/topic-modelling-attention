import numpy as np
import pandas as pd
import scipy.sparse as sp
import ast

import jax
import jax.numpy as jnp
from jax import Array

from nltk.stem import WordNetLemmatizer

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
import seaborn as sns

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


data = fetch_20newsgroups(data_home='./data/', subset='all').data
#data = data[:100]

preprocessor = CorpusLoader(
    min_token_len=1, 
    max_token_len=100, 
    min_df=0.0, 
    max_df=1.0,
    stopwords=set()
)
tokenized_data, document_bounds = preprocessor.fit_transform(data)
print(f'Total number of tokens in preprocessed corpus: {len(document_bounds)}')

'''loader = BatchedCorpusLoader(
    data=tokenized_data,
    doc_bounds=document_bounds,
    batch_size=10000,
)'''
loader = BatchedCorpusLoader(
    data=tokenized_data,
    doc_bounds=document_bounds,
    batch_size=5000,
)
print(f'Number of batches: {len(loader)}')

vocab_size = len(preprocessor.vocabulary)
bow = build_bow(tokenized_data, document_bounds, vocab_size)

td = TopicVarianceMetric(top_k=25, tag="TD@25")
perp = PerplexityMetric()

decorr = DecorrelationRegularization(0.0, "wt")

model = AttentiveTopicModel(
    vocab_size=len(preprocessor.vocabulary),
    ctx_len=100,
    n_topics=100,
    gamma=0.01,
    metrics=[td],
    regularizers=[decorr],
)
'''model = AttentiveTopicModel(
    vocab_size=len(preprocessor.vocabulary),
    ctx_len=20,
    n_topics=20,
    gamma=0.01,
    metrics=[td],
    regularizers=None, #[decorr],
    self_aware_context=True
)'''

model.fit(
    loader,
    max_iter=50,
    verbose=2,
    seed=42,
)

reverse_vocab = {value: key for key, value in preprocessor.vocabulary.items()}
import json
with open("ilya_vocab.json", "w") as f:
    json.dump(reverse_vocab, f, indent=4) 

np.save('ilya_phi.npy', np.asarray(model.phi))