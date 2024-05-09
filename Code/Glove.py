from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json

embeddings_index = {}

def get_cranfield_docs():
	with open('cranfield/cran_docs.json') as f:
		cran_docs = json.load(f)
	return cran_docs

def get_cranfield_queries():
	with open('cranfield/cran_queries.json') as f:
		cran_queries = json.load(f)
	return cran_queries

def process_docs(docs):
	# from the Json, extract the title and body
	processed_docs = []
	for doc in docs:
		processed_docs.append(doc['title'] + ' ' + doc['body'])

	return processed_docs
	
def load_Glove_embeddings_pretrained():
	global embeddings_index

	with open('../glove.6B.50d.txt') as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			embeddings_index[word] = coefs
   
	return embeddings_index

def get_doc_embeddings(docs):
	global embeddings_index

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(docs)
	sequences = tokenizer.texts_to_sequences(docs)
	word_index = tokenizer.word_index
	inverted_word_index = {v: k for k, v in word_index.items()}

	doc_embeddings = []
	for sequence in sequences:
		doc_embedding = [embeddings_index[inverted_word_index[word_id]] for word_id in sequence if inverted_word_index[word_id] in embeddings_index]
		doc_embedding = np.mean(doc_embedding, axis=0)
		doc_embeddings.append(doc_embedding)
	
	return doc_embeddings

if __name__ == '__main__':
	docs = get_cranfield_docs()
	docs = process_docs(docs[:10])

	load_Glove_embeddings_pretrained()
	doc_embeddings = get_doc_embeddings(docs)
	np.save('embeddings/doc_embeddings.npy', doc_embeddings)

	