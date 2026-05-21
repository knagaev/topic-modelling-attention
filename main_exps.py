
import numpy as np
import pandas as pd
import scipy.sparse as sp
import ast
import json
import pickle

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
    get_structured_data
)
from cartm.metrics import (
    PerplexityMetric,
    NPMICoherenceMetric,
    SparsityMetric,
    TopicVarianceMetric,
    StratifiedPerplexityMetric
)
from cartm.regularization import DecorrelationRegularization

from cartm.evaluation.stratified_evaluator import ExternalStratifiedEvaluator

#sns.set_theme()

#from jax import config
#config.update("jax_disable_jit", True)


full_20ng = fetch_20newsgroups(data_home='./data/', subset='all', remove=('headers', 'footers', 'quotes'))
data = full_20ng.data
targets = full_20ng.target
target_names = full_20ng.target_names

#data = fetch_20newsgroups(data_home='./data/', subset='train', categories=categories).data
#data = data[:1000]

corpus_data_prefix = 'new_3_20_10_05_'

data_with_bounds = None
vocab = None
try:
    with open(corpus_data_prefix + 'corpus_data_prefix.pkl', 'rb') as file:
        data_with_bounds = pickle.load(file)
    with open(corpus_data_prefix + 'vocab.json', 'r') as file:
        vocab = json.load(file)
except FileNotFoundError:
    print("corpus_data does not exist, regenerate corpus_data")
    preprocessor = CorpusLoader(min_token_len=3, max_token_len=20, min_df=3, max_df=0.5)
    data_with_bounds = preprocessor.fit_transform(data)
    with open(corpus_data_prefix + 'corpus_data_prefix.pkl', 'wb') as file:
        pickle.dump(data_with_bounds, file)
    vocab = preprocessor.vocabulary
    with open(corpus_data_prefix + "vocab.json", "w") as f:
        json.dump(vocab, f, indent=4) 

tokenized_data, document_bounds = data_with_bounds

print(f'Total number of tokens in preprocessed corpus: {len(tokenized_data)}')
print(f'Total number of documents in preprocessed corpus: {np.sum(document_bounds) + 1}')

vocab_size = len(vocab)
print(f'vocab_size: {vocab_size}')

structure_prefix = 'new_eq_500_50_'
#structure_prefix = 'new_500_50_5_src_'
dataset_structure = {i:500 for i in range(20)}
#dataset_structure[10] = 50 # rec.sport.hockey
#dataset_structure[15] = 50 # soc.religion.christian
test_structure = {i:50 for i in range(20)}
#test_structure[10] = 5
#test_structure[15] = 5

dataset_structures = {
    'full' : {i:1000 for i in range(20)}
}

cfg_n_topics = 30

'''loader = BatchedCorpusLoader(
    data=tokenized_data,
    doc_bounds=document_bounds,
    batch_size=10000,
    #batch_size=100,
)
print(f'Number of batches: {len(loader)}')'''

structured_data = None
try:
    with open(structure_prefix + 'structured_data.pkl', 'rb') as file:
        structured_data = pickle.load(file)
except FileNotFoundError:
    print("structured_data does not exist, regenerate structured_data")
    structured_data = get_structured_data(tokenized_data, document_bounds, targets, 
                dataset_structure, test_structure, order_seed=-1)
    with open(structure_prefix + 'structured_data.pkl', 'wb') as file:
        pickle.dump(structured_data, file)

(train_tokens, train_bounds, train_result_structure), (
 test_tokens, test_bounds, test_result_structure) = structured_data

print(f"{train_result_structure=}")
print(f"{test_result_structure=}")

train_bow = build_bow(train_tokens, train_bounds, vocab_size)
assert train_bow.sum() == len(train_tokens)

perplexity = PerplexityMetric(tag='perplexity')
#sparsity = SparsityMetric(tag='sparsity', eps=1e-8)
#topic_variance = TopicVarianceMetric(top_k=10, tag='topic variance')
npmi_coherence = NPMICoherenceMetric(bow=train_bow, top_k=10, tag='coherence')
stratified_perplexity = StratifiedPerplexityMetric(tag='stratified_perplexity', n_topics=cfg_n_topics)

decorr = DecorrelationRegularization(0.0, "wt")

model = AttentiveTopicModel(
    #self_aware_context=True,
    vocab_size=vocab_size,
    ctx_len=100,
    n_topics=cfg_n_topics,
    gamma=0.01,
    metrics=[perplexity, npmi_coherence, stratified_perplexity],
    regularizers=[]#[decorr],
)

'''model.fit(
    loader,
    max_iter=50,
    verbose=2,
    seed=42,
    num_batches_before_update=1,
    metric_ratio = 0.05,
    save_hist = False
)'''

train_loader = BatchedCorpusLoader(
    data=train_tokens,
    doc_bounds=train_bounds,
    batch_size=10000,
)

test_loader = BatchedCorpusLoader(
    data=test_tokens,
    doc_bounds=test_bounds,
    batch_size=10000,
)

model.fit_with_test(
    train_batches = train_loader,
    test_batches = test_loader,
    max_iter=50,
    verbose=2,
    seed=42,
    num_batches_before_update=1,
    save_hist = False
)

#np.save(filter_mode + "phi_hist.npy", model.phi_hist)
np.save(structure_prefix + "phi.npy", model.phi)

#with open(filter_mode + "metrics.txt", "w") as f:
#    f.write(str(perplexity.history))

metrics = {
    'perplexity': perplexity.history,
    #'sparsity': sparsity.history,
    #'topic_variance': topic_variance.history,
    'npmi_coherence': npmi_coherence.history,
    'stratified_perplexity': stratified_perplexity.history,
    'stratified_perplexity_ppl': stratified_perplexity._last_per_topic
}

with open(structure_prefix + "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4) 
'''
# 3. Инициализация внешнего оценщика (фиксирует phi в текущем состоянии)
evaluator = ExternalStratifiedEvaluator(model, num_attn_passes=1)

# 4. Оценка на валидационном / тестовом наборе
val_results = evaluator.evaluate(train_loader)
print(f"📊 Macro Stratified PPL (val): {val_results['macro_stratified_ppl']:.2f}")
print(f"📈 Documents per topic: {val_results['topic_doc_counts']}")

# 5. Оценка на тестовом наборе (финальный отчёт)
test_results = evaluator.evaluate(test_loader)
print(f"🎯 Macro Stratified PPL (test): {test_results['macro_stratified_ppl']:.2f}")

# 6. Визуализация для отчёта
plot_per_topic_ppl_boxplot(
    final_epoch_data=test_results["per_topic_ppl"],
    topic_labels=[f"T{i}" for i in range(model.n_topics)],
    title="Stratified Per-topic Perplexity (Test Set)",
    output_path="reports/stratified_ppl_test.png"
)'''