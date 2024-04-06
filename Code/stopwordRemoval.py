import nltk

class StopwordRemoval():

	def BottomUpIDF(self, text):
		"""

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed using the Bottom-Up IDF approach
		"""

		# create a list of unique words from the tokenized text
		uniqueWords = list(set([token for sentence in text for token in sentence]))

		# Keep only the words that contain only alphabets, eliminate the ones which contain digits or special characters 
		uniqueWords = [word for word in uniqueWords if word.isalpha()]

		# create a dictionary of words and their IDF values
		idfValues = {}
		for word in uniqueWords:
			idfValues[word] = 0
			for sentence in text:
				if word in sentence:
					idfValues[word] += 1
			idfValues[word] = len(text) / idfValues[word]

		# sort the dictionary by IDF values
		idfValues = sorted(idfValues.items(), key=lambda x: x[1], reverse=True)

		# create a list of stopwords based on threshold IDF value < 1
		stopWordsIDF = [word for word in idfValues if word[1] < 1]
		stopwordRemovedText = [[token for token in sentence if token.lower() not in stopWordsIDF] for sentence in text]

		# write the stop words to a file
		with open('output/stopwords_BottomUpIDF.txt', 'w') as f:
			for word in stopWordsIDF:
				f.write(word + '\n')

		return stopwordRemovedText

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed 
			using the list of stopwords from the nltk library
		"""

		# remove stopwords from the tokenized text using the stopwords list from nltk library
		stopwords = nltk.corpus.stopwords.words('english')
		stopwordRemovedText = [[token for token in sentence if token.lower() not in stopwords] for sentence in text]

		# write the stop words to a file
		with open('output/stopwords_NLTK.txt', 'w') as f:
			for word in stopwords:
				f.write(word + '\n')

		return stopwordRemovedText