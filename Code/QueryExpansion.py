from nltk.corpus import wordnet
class QueryExpansion():
    def append_synonyms(self, token, expanded_sentence):
        expanded_sentence.append(token)

        synsets = wordnet.synsets(token)
        if len(synsets) > 0:
            new_token = synsets[0].name().split('.')[0]
            if new_token != token:
                expanded_sentence.append(new_token)

        return expanded_sentence

    def expansion(self, queries):
        expanded_queries = []

        for query in queries:
            expanded_query = []

            for sentence in query:
                expanded_sentence = []

                for token in sentence:
                    token = token.lower()

                    if token.isalpha() == True:
                        expanded_sentence = self.append_synonyms(token, expanded_sentence)

                    else:
                        if '-' in token:
                            expanded_sentence.append(token)

                            synsets = wordnet.synsets(token)
                            if len(synsets) > 0:
                                new_token = synsets[0].name().split('.')[0]
                                if new_token != token:
                                    expanded_sentence.append(new_token)
                        
                            else:
                                tokens = token.split('-')
                                for w in tokens:
                                    expanded_sentence = self.append_synonyms(w, expanded_sentence)
            
                expanded_query.append(expanded_sentence)

            expanded_queries.append(expanded_query)

        return expanded_queries
    