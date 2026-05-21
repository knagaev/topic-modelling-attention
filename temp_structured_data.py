import pickle

CATEGORIES_QTY = 20
CATEGORY_MED = 13 # sci.med
CATEGORY_REL = 15 # soc.religion.christian
CORPUS_FILENAME = 'corpus_data.pkl'
STRUCTURED_DATA_FILENAME = 'structured_data.pkl'
VOCAB_FILENAME = 'corpus_vocab.json'

CORPUS_PATH = 'data/corpuses/'
STRUCTURED_DATA_PATH = 'data/structured/'
structured_data_prefix = 'full'

structured_data_filepath = CORPUS_PATH + structured_data_prefix + '_' + STRUCTURED_DATA_FILENAME

with open(structured_data_filepath, 'rb') as file:
    structured_data = pickle.load(file)
    print('structured_data ok')

(train_tokens, train_bounds, train_result_structure), (
 test_tokens, test_bounds, test_result_structure) = structured_data

print(f"{train_result_structure=}")
print(f"{test_result_structure=}")