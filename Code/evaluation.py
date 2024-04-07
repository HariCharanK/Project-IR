from util import *
import numpy as np

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		s = set(true_doc_IDs)
		precision = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in s:
				precision += 1
		
		precision /= k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[i['query_num']].append(i['id'])

		meanPrecision = 0
		for i in range(len(query_ids)):
			meanPrecision += self.queryPrecision(self,doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)

		meanPrecision /= len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		s = set(true_doc_IDs)
		recall = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in s:
				recall += 1
		
		recall /= len(s)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[i['query_num']].append(i['id'])

		meanRecall = 0
		for i in range(len(query_ids)):
			meanRecall += self.queryRecall(self,doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)

		meanRecall /= len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		precision = self.queryPrecision(self,query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(self,query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = 0

		if precision + recall != 0:
			fscore = 2 * precision * recall / (precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[i['query_num']].append(i['id'])
		
		meanFscore = 0
		for i in range(len(query_ids)):
			meanFscore += self.queryFscore(self,doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)
		
		meanFscore /= len(query_ids)
		return meanFscore
	


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		
		DCG = 0
		IDCG = 0
		query_relevance = {}
		for i in qrels:
			if i['query_num'] == query_id:
				query_relevance[i['id']] = i['position']
		for i in range(k):
			if query_doc_IDs_ordered[i] in query_relevance:
				DCG += query_relevance[query_doc_IDs_ordered[i]] / np.log2(i + 2)
		
		for i in range(k):
			if true_doc_IDs[i] in query_relevance:
				IDCG += query_relevance[true_doc_IDs[i]] / np.log2(i + 2)
	
		return DCG / IDCG
	


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0
		for i in range(len(query_ids)):
			meanNDCG += self.queryNDCG(doc_IDs_ordered[i], query_ids[i], self.getTrueDocIDs(qrels, query_ids[i]), qrels, k)
		meanNDCG /= len(query_ids)
		return meanNDCG



	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0
		for i in range(1,k+1):
			avgPrecision += self.queryPrecision(self,query_doc_IDs_ordered, query_id, true_doc_IDs, i)
		avgPrecision /= k

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[i['query_num']].append(i['id'])
		
		meanAveragePrecision = 0
		for i in range(len(query_ids)):
			meanAveragePrecision += self.queryAveragePrecision(self,doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)
		
		meanAveragePrecision /= len(query_ids)

		return meanAveragePrecision

 
