import pickle
import sys
import json

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


CATEGORIES_QTY = 20
CATEGORY_MED = 13 # sci.med
CATEGORY_REL = 15 # soc.religion.christian
CORPUS_FILENAME = 'corpus_data.pkl'
STRUCTURED_DATA_FILENAME = 'structured_data.pkl'
VOCAB_FILENAME = 'corpus_vocab.json'

CORPUS_PATH = 'data/corpuses/'
STRUCTURED_DATA_PATH = 'data/structured/'
RESULT_PATH = 'data/results/'

#if len(sys.argv) < 3:
#    print('No args, program STRUCTURED_DATA_PREFIX EXPMODE')
#    exit()

corpus_data_prefix = 'full'
#structured_data_prefix = 'full'
structured_data_prefix = sys.argv[1] # датасет ['full', 'imbalanced_test', 'minus_rel', 'minus_med']
#exp_mode = 'base'
exp_mode = sys.argv[2] # режим эксперимента
print(f"{structured_data_prefix=}")
print(f"{exp_mode=}")


structured_data_filepath = STRUCTURED_DATA_PATH + structured_data_prefix + '_' + STRUCTURED_DATA_FILENAME

with open(structured_data_filepath, 'rb') as file:
    structured_data = pickle.load(file)
    print('structured_data ok')

(train_tokens, train_bounds, train_result_structure), (
 test_tokens, test_bounds, test_result_structure) = structured_data

print(f"{train_result_structure=}")
print(f"{test_result_structure=}")

vocab_filepath = CORPUS_PATH + corpus_data_prefix + '_' + VOCAB_FILENAME
with open(vocab_filepath, 'r') as file:
    vocab = json.load(file)
    vocab_size = len(vocab)
    #reverse_vocab = {value: key for key, value in vocab.items()}

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

experiments = {
    'base': {    
        'n_topics': [20],
        'gamma': [0.01],
        'ctx_len': [100],
    },
    'gamma': {
        'n_topics': [20],
        'gamma': [0.005, 0.05, 0.1, 0.5],
        'ctx_len': [100],
    },
    'n_topics_gamma': {
        'n_topics': [10, 50, 70, 100],
        'gamma': [0.005, 0.05, 0.1, 0.5],
        'ctx_len': [100],
    },
    'n_topics_gamma': {
        'n_topics': [10, 20, 30, 50, 70, 100],
        'gamma': [0.005, 0.05, 0.01, 0.1, 0.5],
        'ctx_len': [100],
    }
}

exp_config = experiments[exp_mode]

for n_topics in exp_config['n_topics']:
    for gamma in exp_config['gamma']:
        for ctx_len in exp_config['ctx_len']:

            perplexity = PerplexityMetric(tag='perplexity')
            #sparsity = SparsityMetric(tag='sparsity', eps=1e-8)
            #topic_variance = TopicVarianceMetric(top_k=10, tag='topic variance')
            train_bow = build_bow(train_tokens, train_bounds, vocab_size)
            assert train_bow.sum() == len(train_tokens)
            npmi_coherence = NPMICoherenceMetric(bow=train_bow, top_k=10, tag='coherence')
            stratified_perplexity = StratifiedPerplexityMetric(tag='stratified_perplexity', n_topics=n_topics)

            metrics = {
                'perplexity': perplexity.history,
                'npmi_coherence': npmi_coherence.history,
                'stratified_perplexity': stratified_perplexity.history,
                'stratified_pt_perplexity': stratified_perplexity.per_topic_history
            }


            model = AttentiveTopicModel(
                #self_aware_context=True,
                vocab_size=vocab_size,
                ctx_len=ctx_len,
                n_topics=n_topics,
                gamma=gamma,
                metrics=[perplexity, npmi_coherence, stratified_perplexity],
                regularizers=[]#[decorr],
            )

            print('n_topics_{n_topics}_gamma_{gamma}_ctx_len_{ctx_len}_{structured_data_prefix}'.format(n_topics=n_topics, gamma=gamma, ctx_len=ctx_len, structured_data_prefix=structured_data_prefix))

            model.fit_with_test(
                train_batches = train_loader,
                test_batches = test_loader,
                max_iter=50,
                verbose=2,
                seed=42,
                num_batches_before_update=1,
                save_hist = False
            )

            #result_filename = 'result_n_topics_' + str(n_topics) + '_gamma_' + str(gamma + 'ctx_len' + ctx_len + '.json'
            result_filename = 'result_n_topics_{n_topics}_gamma_{gamma}_ctx_len_{ctx_len}_{structured_data_prefix}.json'.format(n_topics=n_topics, gamma=gamma, ctx_len=ctx_len, structured_data_prefix=structured_data_prefix)
            result_filepath = RESULT_PATH + result_filename

            with open(result_filepath, "w") as f:
                json.dump(metrics, f, indent=4) 
