import jax
import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns
from sys import path as syspath

syspath.insert(1, r"D:\ARTM\topic-modelling-attention\src")

from cartm.model import ContextTopicModel
from cartm.preprocessing import DatasetPreprocessor


data = fetch_20newsgroups(data_home="./data/", subset="all").data
print(f"Total number of documents in corpus: {len(data)}")
print(f"Total number of words in corpus: {sum([len(doc.split(' ')) for doc in data])}")

print("Ok")
