import nltk
import collections
import operator

nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def createChunkingEn(string):
    tokens = nltk.word_tokenize(string)
    tags = nltk.pos_tag(tokens)
    chunkingEn = nltk.ne_chunk(tags, binary = True)

    return chunkingEn

def createWordList(tree):
    for x in chunkingEn.subtrees():
     if hasattr(x, 'label'):
         if x.label() == "NE":
             words = [w[0] for w in x.leaves()]

    return words

def createRanking(words):
    counterWords = dict(collections.Counter(words))
    sortedCounterWords = sorted(counterWords.items(), key = operator.itemgetter(1))
    sortedCounterWords = list(reversed(sortedCounterWords))

    return sortedCounterWords

def deleteMostFrecuencyWord(word, string):
    return string.replace(word, "")

string = 'Prince William and his new new wife Catherine new kissed twice to satisfy the besotted Buckingham Palace crowds'

chunkingEn = createChunkingEn(string)
wordsList = createWordList(chunkingEn.subtrees())
rankingWords = createRanking(wordsList)
print(rankingWords)
