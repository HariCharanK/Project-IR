from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

typo_words = ['boundery', 'transiant', 'aerplain'] 
doc_path = 'output/tokenized_docs.txt'
vocabulary = []

class SpellCheck:

    def build_vocabulary(self, doc_path):
        """Constructs the vocabulary (V) from all documents."""

        global vocabulary

        text = ""
        with open(doc_path, 'r') as f:
            text = f.read().splitlines()[0]
        
        text = eval(text)

        for doc in text:
            for sentence in doc:
                for token in sentence:
                    if token.isalpha():
                        vocabulary.append(token)
        
        vocabulary = list(set(vocabulary))
        vocabulary.sort()

    def create_bigram_vector(self, token):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        all_bigrams = [a + b for a in alphabet for b in alphabet]
        vector = np.zeros(len(all_bigrams))
        for i, bigram in enumerate(all_bigrams):
            vector[i] = token.count(bigram)
        return vector

    def find_top_corrections(self, typo, n=5):
        """Finds the top N candidate corrections for a given typo."""
        
        global vocabulary

        candidates = defaultdict(float)
        typo_vector = self.create_bigram_vector(typo)

        for word in vocabulary:
            word_vector = self.create_bigram_vector(word)

            # Calculate similarity using cosine similarity
            similarity = cosine_similarity([typo_vector], [word_vector])[0][0]

            candidates[word] = similarity

        # Sort candidates by similarity and return top N
        return sorted(candidates.items(), key=lambda item: item[1], reverse=True)[:n] 

    def EditDistance(self, str1, str2, insert_cost=1, del_cost=1, sub_cost=1):
        m = len(str1) + 1  
        n = len(str2) + 1  

        dp = [[0 for i in range(n)] for j in range(m)]

        for i in range(1, m):
            dp[i][0] = i * del_cost  

        for j in range(1, n):
            dp[0][j] = j * insert_cost

        for i in range(1, m):
            for j in range(1, n):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + del_cost, dp[i][j-1] + insert_cost,  dp[i-1][j-1] + sub_cost)  

        return dp[m-1][n-1]
    

# top 5 candidate corrections corresponding to each typo
spellCheck = SpellCheck()
spellCheck.build_vocabulary(doc_path)

top_candidates = {}
for typo in typo_words:
    print("Top 5 candidate corrections for '{}':".format(typo))

    candidate_corrections =  spellCheck.find_top_corrections(typo)
    candidates = [candidate_corrections[i][0] for i in range(5)]

    for candidate in candidates:
        print(candidate)

    # candidate closest to the typo using Edit Distance 
    print("\nCandidate closest to the typo using Edit Distance : \n", 
          min(candidates, key=lambda candidate: spellCheck.EditDistance(typo, candidate)))

    print("\n")
