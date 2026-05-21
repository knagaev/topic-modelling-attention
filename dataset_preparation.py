import sys
import json
import pickle

from sklearn.datasets import fetch_20newsgroups
from cartm.preprocessing import CorpusLoader, get_structured_data

CATEGORIES_QTY = 20
CATEGORY_MED = 13 # sci.med
CATEGORY_REL = 15 # soc.religion.christian

BIG_TRAIN_QTY = 1000
BIG_TRAIN_QTY = 1000
SMALL_CATEGORY_QTY = 50

CORPUS_FILENAME = 'corpus_data.pkl'
STRUCTURED_DATA_FILENAME = 'structured_data.pkl'
VOCAB_FILENAME = 'corpus_vocab.json'

CORPUS_PATH = 'data/corpuses/'
STRUCTURED_DATA_PATH = 'data/structured/'

config_structures = {
    'full' : {
        'dataset_structure': {i:1000 for i in range(CATEGORIES_QTY)},
        'test_structure': {i:50 for i in range(CATEGORIES_QTY)},
    },
    'imbalanced_test' : {
        'dataset_structure': {i:1000 for i in range(CATEGORIES_QTY)},
        'test_structure': {i:(10 if i == CATEGORY_REL else 50) for i in range(CATEGORIES_QTY)},
    },
    'minus_rel' : {
        'dataset_structure': {i:(50 if i == CATEGORY_REL else 1000) for i in range(CATEGORIES_QTY)},
        'test_structure': {i:(10 if i == CATEGORY_REL else 50) for i in range(CATEGORIES_QTY)},
    },
    'minus_med' : {
        'dataset_structure': {i:(50 if i == CATEGORY_MED else 1000) for i in range(CATEGORIES_QTY)},
        'test_structure': {i:(10 if i == CATEGORY_MED else 50) for i in range(CATEGORIES_QTY)},
    },
}


if len(sys.argv) < 3:
    print('No args, program CORPUS_FILENAME STRUCTURED_DATA_FILENAME')
    exit()

corpus_prefix = sys.argv[1]
structured_data_prefix = sys.argv[2]
print(f"{corpus_prefix=}")
print(f"{structured_data_prefix=}")

full_20ng = fetch_20newsgroups(data_home='./data/', subset='all', remove=('headers', 'footers', 'quotes'))
data = full_20ng.data
targets = full_20ng.target
target_names = full_20ng.target_names

corpus_filepath = CORPUS_PATH + corpus_prefix + '_' + CORPUS_FILENAME
structured_data_filepath = STRUCTURED_DATA_PATH + structured_data_prefix + '_' + STRUCTURED_DATA_FILENAME

try:
    with open(structured_data_filepath, 'rb') as file:
        structured_data = pickle.load(file)
        print('structured_data ok')
        (train_tokens, train_bounds, train_result_structure), (
                test_tokens, test_bounds, test_result_structure) = structured_data
        print(f"{train_result_structure=}")
        print(f"{test_result_structure=}")

except FileNotFoundError:
    print("structured_data does not exist, regenerate structured datafile")
    with open(corpus_filepath, 'rb') as file:
        data_with_bounds = pickle.load(file)
        print('corpus data ok')
        tokenized_data, document_bounds = data_with_bounds
    dataset_structure = config_structures[structured_data_prefix]['dataset_structure']
    test_structure = config_structures[structured_data_prefix]['test_structure']
    structured_data = get_structured_data(tokenized_data, document_bounds, targets, 
                dataset_structure, test_structure, order_seed=-1)

    (train_tokens, train_bounds, train_result_structure), (
        test_tokens, test_bounds, test_result_structure) = structured_data

    print(f"{train_result_structure=}")
    print(f"{test_result_structure=}")

    with open(structured_data_filepath, 'wb') as file:
        pickle.dump(structured_data, file)
