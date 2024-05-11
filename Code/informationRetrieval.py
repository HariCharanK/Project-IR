from util import *


# Add your import statements here

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable
		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""
		self.count_vectorizer = CountVectorizer()
		# flattened_documents = [' '.join(sentence) for document in docs for sentence in document]

		# # Generate document-term matrix
		# dtm = self.count_vectorizer.fit_transform(flattened_documents)
		# vocabulary = self.count_vectorizer.get_feature_names_out()

		# document_index = {}
		# # Loop through each document
		# documents = docs
		# for i, doc_id in enumerate(docIDs):
		# 	document_sentences = [sentence for sentence in documents[i]]
		# 	document_vectors = []
		# 	for sentence in document_sentences:
		# 		sentence_vector = self.count_vectorizer.transform([' '.join(sentence)]).toarray().flatten()
		# 		document_vectors.append(sentence_vector)
		# 	document_vector = sum(document_vectors)
		# 	for j, term in enumerate(vocabulary):
		# 		if document_vector[j] > 0:
		# 			if term not in document_index:
		# 				document_index[term] = []
		# 			document_index[term].append(doc_id)
		
		flattened_documents = [' '.join(list(chain.from_iterable(x))) for x in docs]
		self.term_doc_freq = self.count_vectorizer.fit_transform(flattened_documents)
		self.docIDs = docIDs
		self.index = self.term_doc_freq.T



	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query
		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		
		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		term_doc_freq = self.term_doc_freq
		self.tfidf_transformer = TfidfTransformer(norm='l2',use_idf=True, smooth_idf=False,sublinear_tf=False)
		self.tfidf_transformer.fit(term_doc_freq)
		tfidf = self.tfidf_transformer.transform(term_doc_freq)


		flattened_queries = [' '.join(list(chain.from_iterable(x))) for x in queries]
		querycounts = self.count_vectorizer.transform(flattened_queries)
		querytfidf = self.tfidf_transformer.transform(querycounts)

		cossimilarity = cosine_similarity(querytfidf,tfidf)
		docIDS = self.docIDs

		for cos_similarity_vector in cossimilarity:
			top_doc_indices = cos_similarity_vector.argsort()[::-1]
			top_docs = [docIDS[docidx] for docidx in top_doc_indices]
			doc_IDs_ordered.append(top_docs)

	
		return doc_IDs_ordered

