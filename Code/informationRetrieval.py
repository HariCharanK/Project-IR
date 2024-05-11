import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class InformationRetrieval():

    def __init__(self):
        self.doc_ids = None
        self.vectorizer = None
        self.svd = None
        self.reduced_matrix = None

    def buildIndex(self, documents, document_ids, ngram, concepts):
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

        all_docs_combined = []
        for document in documents:
            all_sentences_combined = []
            for sentence in document:
                all_sentences_combined.extend(sentence)
            all_docs_combined.append(all_sentences_combined)
        all_docs_combined = [' '.join(sentence) for sentence in all_docs_combined]
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        term_document_matrix = tfidf_vectorizer.fit_transform(all_docs_combined)
        svd_model = TruncatedSVD(random_state=42, n_components=concepts)
        reduced_matrix = svd_model.fit_transform(term_document_matrix)

        self.doc_ids = document_ids
        self.vectorizer = tfidf_vectorizer
        self.svd = svd_model
        self.reduced_matrix = reduced_matrix

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

        all_queries_combined = []
        for query in queries:
            all_sentences_combined = []
            for sentence in query:
                all_sentences_combined.extend(sentence)
            all_queries_combined.append(all_sentences_combined)
        all_queries_combined = [' '.join(sentence) for sentence in all_queries_combined]

        query_vectorizer = self.vectorizer.transform(all_queries_combined)
        query_vectors = self.svd.transform(query_vectorizer)

        similarity_values = cosine_similarity(query_vectors, self.reduced_matrix)
        ranked_doc_ids = []
        for value in similarity_values:
            scores = []
            total_docs = len(self.doc_ids)
            for i in range(total_docs):
                scores.append((self.doc_ids[i], value[i]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            ordered_docs_query = []
            for i in sorted_scores:
                ordered_docs_query.append(i[0])
            ranked_doc_ids.append(ordered_docs_query)
        return ranked_doc_ids
