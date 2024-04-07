from util import *
import math



class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.idf = None

	def eval_idf(self,docsIDs):
		idf = {}
		for i in docsIDs:
			for j in self.index[i]:
				if j in idf:
					idf[j] += 1
				else:
					idf[j] = 1

		N = len(docsIDs)
		for i in idf:
			idf[i] = idf[i]/N
			idf[i] = math.log(1/idf[i])

		self.idf = self.idf

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

		index = {}
		for i in range(len(docs)):
			m ={}
			for j in docs[i]:
				for k in j:
					if k in m:
						m[k] += 1
					else:
						m[k] = 1
			index[docIDs[i]] = m
		
		self.index = index
		self.eval_idf(docIDs)


	def cosine_sim(self,temp,doc_vector):
		dot_product = 0
		mag_query = 0
		mag_doc = 0
		for i in temp:
			if i in doc_vector:
				dot_product += temp[i]*doc_vector[i]*self.idf[i]
			mag_query += temp[i]**2
		for i in doc_vector:
			mag_doc += doc_vector[i]**2
		mag_query = math.sqrt(mag_query)
		mag_doc = math.sqrt(mag_doc)
		sim = dot_product/(mag_query*mag_doc)
		return sim


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

		docIDs = list(self.index.keys())
		doc_IDs_ordered = []

		for i in range(len(queries)):
			#building the tf-idf vector for query
			query = sum(query,[])
			temp = {}
			for word in query:
				if word in temp:
					temp[word]+=1
				else:
					temp[word]=1
			for i in temp:
				if i in self.idf:
					temp[i] = temp[i]*self.idf[i]
			
			#computing cosine-sim with each doc
			m={}
			for doc in docIDs:
				doc_vector = self.index[doc]
				sim = self.cosine_sim(temp,doc_vector)
				m[doc] = sim
			
			#sorting the docs based on cosine-sim
			m = dict(sorted(m.items(), key=lambda item: item[1],reverse=True))
			doc_IDs_ordered.append(list(m.keys()))

	
		return doc_IDs_ordered




