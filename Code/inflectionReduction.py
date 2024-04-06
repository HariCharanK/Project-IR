import nltk
print("NLTK Data Path: \n", nltk.data.path)

nltk.download('stopwords')
print("downloaded stopwords\n\n")

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = []

		# reduce the inflection of the tokenized-text using the Porter Stemmer
		stemmer = nltk.stem.PorterStemmer()
		for sentence in text:
			reducedText.append([stemmer.stem(token.lower()) for token in sentence])
		
		return reducedText