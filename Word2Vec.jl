import Pkg
Pkg.add("Word2Vec")

import Word2Vec
using Word2Vec

#Word2Vec

word2vec("Cronista.txt", "Cronista-vec.txt", verbose = true, min_count=1)
model = wordvectors("./Cronista-vec.txt")




word2phrase("Cronista.txt", "CronistaPhrase.txt")
word2vec("CronistaPhrase.txt", "CronistaPhrase-vec.txt", verbose=true)
model2 = wordvectors("CronistaPhrase-vec.txt")


get_vector(model2, "Alberto_Fernández")
cosine_similar_words(model2, "Alberto_Fernández")

words = vocabulary(model2)
idx = index(model2, "Alberto_Fernández")
words[idx]

# We can retrieve the vector representation of individual words and compute the cosine distance between two words.

w2v_1 = similarity(model2, "Alberto_Fernández", "inflación")


# The funciton cosine(model, word, n) return the indices and distances of n neighbors of word. 
idxs, dists = cosine(model2, "Alberto_Fernández", 20)

# We can use Gadfly to plot the top 10 similar words 
Pkg.add("Gadfly")
import Gadfly
using Gadfly
plot(x=words[idxs], y=dists)

"""
analogy(wv, pos, neg, n=5)
Compute the analogy similarity between two lists of words. The positions and the similarity values of the top n similar words will be returned. For example,  king - man + woman = queen will be  pos=["king", "woman"], neg=["man"].
"""

indxs, dists = analogy(model, ["Alberto"], ["Madero"], 8)
plot(x=words[indxs], y=dists)

"""
analogy_words(wv, pos, neg, n=5)
Return the top n words computed by analogy similarity between positive words pos and negaive words neg. from the WordVectors wv.
"""

analogy_words(model, ["Alberto"], ["Fernández"], 10)



