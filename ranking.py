import nltk
import collections
import operator

nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

string = 'Prince William and his new new wife Catherine new kissed twice to satisfy the besotted Buckingham Palace Palace crowds'
tokens = nltk.word_tokenize(string)
# print(tokens)

tags = nltk.pos_tag(tokens)
# print(tags)

chunkingEn = nltk.ne_chunk(tags, binary = True)
# print(chunkingEn)

for x in chunkingEn.subtrees():
 if hasattr(x, 'label'):
     if x.label() == "NE":
         words = [w[0] for w in x.leaves()]
         name = " " . join(words)
         # print(name)

counterWords = dict(collections.Counter(words))

sortedCounterWords = sorted(counterWords.items(), key = operator.itemgetter(1))
sortedCounterWords = list(reversed(sortedCounterWords))
print(sortedCounterWords)
