import jax
import jax.numpy as jnp

from sklearn.datasets import fetch_20newsgroups

import matplotlib.pyplot as plt
import seaborn as sns

from cartm.model import ContextTopicModel
from cartm.preprocessing import DatasetPreprocessor

import os 

def test():
    jax.config.update('jax_num_cpu_devices', 8)

    data = ["From: Mamatha Devineni Ratnam <mr47+@andrew.cmu.edu>\nSubject: Pens fans reactions\nOrganization: Post Office, Carnegie Mellon, Pittsburgh, PA\nLines: 12\nNNTP-Posting-Host: po4.andrew.cmu.edu\n\n\n\nI am sure some bashers of Pens fans are pretty confused about the lack\nof any kind of posts about the recent Pens massacre of the Devils. Actually,\nI am  bit puzzled too and a bit relieved. However, I am going to put an end\nto non-PIttsburghers' relief with a bit of praise for the Pens. Man, they\nare killing those Devils worse than I thought. Jagr just showed you why\nhe is much better than his regular season stats. He is also a lot\nfo fun to watch in the playoffs. Bowman should let JAgr have a lot of\nfun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final\nregular season game.          PENS RULE!!!\n\n", 'From: mblawson@midway.ecn.uoknor.edu (Matthew B Lawson)\nSubject: Which high-performance VLB video card?\nSummary: Seek recommendations for VLB video card\nNntp-Posting-Host: midway.ecn.uoknor.edu\nOrganization: Engineering Computer Network, University of Oklahoma, Norman, OK, USA\nKeywords: orchid, stealth, vlb\nLines: 21\n\n  My brother is in the market for a high-performance video card that supports\nVESA local bus with 1-2MB RAM.  Does anyone have suggestions/ideas on:\n\n  - Diamond Stealth Pro Local Bus\n\n  - Orchid Farenheit 1280\n\n  - ATI Graphics Ultra Pro\n\n  - Any other high-performance VLB card\n\n\nPlease post or email.  Thank you!\n\n  - Matt\n\n-- \n    |  Matthew B. Lawson <------------> (mblawson@essex.ecn.uoknor.edu)  |   \n  --+-- "Now I, Nebuchadnezzar, praise and exalt and glorify the King  --+-- \n    |   of heaven, because everything he does is right and all his ways  |   \n    |   are just." - Nebuchadnezzar, king of Babylon, 562 B.C.           |   \n']
    print(f'Total number of documents in corpus: {len(data)}')
    print(f'Total number of words in corpus: {sum([len(doc.split(' ')) for doc in data])}')

    preprocessor = DatasetPreprocessor()
    tokenized_data, document_bounds = preprocessor.fit_transform(data)
    print(f'Total number of document boundaries in preprocessed corpus: {len(document_bounds)}')
    print(f'Total number of tokenized words in preprocessed corpus: {len(tokenized_data)}')

    topic_model = ContextTopicModel(
        vocab_size=len(preprocessor.vocabulary),
        ctx_len=10,
    )

    topic_model.fit(
        data=tokenized_data,
        ctx_bounds=document_bounds,
        verbose=2,
        seed=42,
    )

if __name__ == "__main__":
    test()
