from __future__ import division
import nltk
# nltk.download()
# from nltk.book import *
# nltk.download('omw')

# Calculate lexical diversity
def lexical_diversity(text):
    return len(text) / len(set(text))

# Try to detect plural form of English noun.
def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

# Compute vocabulary of a text, then filter to just uncommon/misspelled words.
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

# Compute what fraction of words in a text are not in the stopwords list
# Use a lexical resource to filter content of a text corpus
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)
# print(content_fraction(nltk.corpus.reuters.words()))

# Compute all variations of puzzle_letters that use 'r'
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
#print([w for w in wordlist if len(w) >= 6 and obligatory in w and nltk.FreqDist(w) <= puzzle_letters])

# Find gender-ambiguous names
names = nltk.corpus.names
names.fileids()
['female.txt', 'male.txt']
male_names = names.words('male.txt')
female_names = names.words('female.txt')
# print([w for w in male_names if w in female_names])
# print(len([w for w in male_names if w in female_names]))

# CFD plot to show counts of name endings
cfd = nltk.ConditionalFreqDist( (fileid, name[-1])
    for fileid in names.fileids()
    for name in names.words(fileid))
# cfd.plot()
