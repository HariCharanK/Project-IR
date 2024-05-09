import os
import sys
dir_path_local = os.path.dirname(__file__)
# adding the source directory to the PYTHONPATH env variable
sys.path.append(os.path.join(dir_path_local, '..'))

import numpy as np
from Glove import *

def test_precomputed_embeddings():
    doc_embeddings = np.load(os.path.join(dir_path_local, '../embeddings/doc_embeddings.npy'))
    assert doc_embeddings.shape == (1400, 50)

    query_embeddings = np.load(os.path.join(dir_path_local, '../embeddings/query_embeddings.npy'))    
    assert query_embeddings.shape == (225, 50)

def test_get_doc_embeddings():
    embeddings_index = load_Glove_embeddings_pretrained()

    docs = ['test', 'hello']
    doc_embeddings = get_doc_embeddings(docs)

    for i in range(len(docs)):
        assert np.array_equal(doc_embeddings[i], embeddings_index[docs[i]])
        