import sys
import json
import pickle

from sklearn.datasets import fetch_20newsgroups
from cartm.preprocessing import CorpusLoader

CORPUS_FILENAME = 'corpus_data.pkl'
VOCAB_FILENAME = 'corpus_vocab.json'
CORPUS_PATH = 'data/corpuses/'

if len(sys.argv) < 2:
    print('No corpus_data_prefix')
    exit()

corpus_data_prefix = sys.argv[1]
print(f"{corpus_data_prefix=}")

full_20ng = fetch_20newsgroups(data_home='./data/', subset='all', remove=('headers', 'footers', 'quotes'))
data = full_20ng.data
targets = full_20ng.target
target_names = full_20ng.target_names

corpus_filepath = CORPUS_PATH + corpus_data_prefix + '_' + CORPUS_FILENAME
vocab_filepath = CORPUS_PATH + corpus_data_prefix + '_' + VOCAB_FILENAME

try:
    with open(corpus_filepath, 'rb') as file:
        data_with_bounds = pickle.load(file)
        print('corpus ok')
    with open(vocab_filepath, 'r') as file:
        vocab = json.load(file)
        print('vocab ok')
except FileNotFoundError:
    print("corpus_data does not exist, regenerate corpus datafile")
    preprocessor = CorpusLoader(min_token_len=3, max_token_len=20, min_df=3, max_df=0.5)
    data_with_bounds = preprocessor.fit_transform(data)

    tokenized_data, document_bounds = data_with_bounds
    print(f'Total number of tokens in preprocessed corpus: {len(tokenized_data)}')
    print(f'Total number of documents in preprocessed corpus: {sum(document_bounds) + 1}')

    with open(corpus_filepath, 'wb') as file:
        pickle.dump(data_with_bounds, file)

    vocab = preprocessor.vocabulary
    with open(vocab_filepath, "w") as f:
        json.dump(vocab, f, indent=4) 
