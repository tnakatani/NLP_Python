# Natural Language Processing Notes

## NLTK Basics

### Setting Up

```python
import nltk
nltk.download()
from nltk.book import *
```

## Technical Terms
1. **Token** is a sequence of characters treated as a group (ie. words).
2. **Word Type** is the form or spelling of the word independently of its specific occurrences in a text.
3. **Hapaxes** are words that occur only once.
4. **Collocation** sequence of words that occur together unusually often.  Resistant to substitution with words that have similar sense (e.g. 'red wine' vs 'maroon wine')
5. **Bigrams** is a list of word pairs.  Word groups are **N-grams**.
6. **Text alignment** - pair up sentences between a bilingual document to detect crresponding words/phrases and build translation model.
7. **Stylistics** is a study of systematic differences between genres.
8. Lexical Resource is a collection of words and/or phrases along with associated information, such as part-of-speech and sense definitions. Lexical resources are secondary to texts, and are usually created and enriched with the help of texts.

## Statistical Language Models

### Frequency Distribution

**Frequency Distribution** tells the frequency of each word in the text.

```python
fdist1 = FreqDist(text1)
fdist1
# <FreqDist with 260819 outcomes>

vocabulary1 = fdist1.keys() >>> vocabulary1[:50]
# [',', 'the', '.', 'of', 'and', 'a', 'to', ';', 'in', 'that', "'", '-', 'his', 'it', 'I', 's', 'is', 'he', 'with', 'was', 'as', '"', 'all', 'for', 'this', '!', 'at', 'by', 'but', 'not', '--', 'him', 'from', 'be', 'on', 'so', 'whale', 'one', 'you', 'had', 'have', 'there', 'But', 'or', 'were', 'now', 'which', '?', 'me', 'like']

fdist1['whale']
# 906
```

#### Finding Specific Words
This means “the set of all w such that w is an element of V (the vocabulary) and w has property P.”

1. {w|w ∈ V & P(w)}
2. [w for w in V if p(w)]

##### For each word w in the vocabulary V, check whether len(w) is greater than 15; all other words will be ignored.
```python
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

# ['CIRCUMNAVIGATION', 'Physiognomically', 'apprehensiveness', 'cannibalistically', 'characteristically', 'circumnavigating', 'circumnavigation', 'circumnavigations', 'comprehensiveness', 'hermaphroditical', 'indiscriminately', 'indispensableness', 'irresistibleness', 'physiognomically', 'preternaturalness', 'responsibilities', 'simultaneousness', 'subterraneousness', 'supernaturalness', 'superstitiousness', 'uncomfortableness', 'uncompromisedness', 'undiscriminating', 'uninterpenetratingly']
```

##### Search words longer than seven characters, that occur more than seven times
```python
fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])
```

##### Find words that contain 'cie' or 'cei'
```python
tricky = sorted([w for w in set(text2) if 'cie' in w or 'cei' in w])
for word in tricky:
  print(word)
```


#### Finding Patterns (Other Than Words)

##### Distribution of word lengths in a text
```python
[len(w) for w in text1]
#[1, 4, 4, 2, 6, 8, 4, 1, 9, 1, 1, 8, 2, 1, 4, 11, 5, 2, 1, 7, 6, 1, 3, 4, 5, 2, ...]

fdist = FreqDist([len(w) for w in text1])
fdist
#<FreqDist with 260819 outcomes>

fdist.keys()
#[3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
```

---

## Automatic Natural Language Understanding

### Word Sense Disambiguation

Understand which sense of the word was intended in a given context (e.g. 'serve' as in food, or baseball?.  'Dish' as in plate or satellite?)

#### Contexts
1. The lost children were found by the searchers (**agentive**)
2. The lost children were found by the mountain (**locative**)
3. The lost children were found by the afternoon (**temporal**)

### Pronoun Resolution

"Who did what to whom", ie. how to detect subjects and objects of verbs.

#### Computational techniques:
1. **Anaphora resolution** - identify what a pronoun or noun phrase refers to
2. **Semantic Role Labeling** - idnetify how a noun phrase relates to the verb

### Spoken Dialogue Systems
![pipeline](./img/pipeline_architecture.png)

### Textual Entailment
Making an inference on a hypothesis given a reference text.

Example:
1. Text: David Golinkin is the editor or author of 18 books, and over 150 responsa, articles, sermons and books
2. Hypothesis: Golinkin has written 18 books

In order for system to understand, it needs to know that:
1. if someone is an author of a book, then he/ she has written that book;
2. if someone is an editor of a book, then he/she has not written (all of) that book;
3. if someone is editor or author of 18 books, then one cannot conclude that he/she is author of 18 books.

---

## Accessing Text Corpora and Lexical Resources

**Corpora** is a large body of linguistic data.

Corpora Examples in NLTK:
1. Gutenberg
2. Web and Chat Text
3. Brown Corpus - categorized by genre (news, editorial, religion)
4. Reuters Corpus - classified into 90 topics, grouped into two ("trainig", "test") for training/testing topic detection algorithms.
5. Inaugural Address Corpus
6. Annotated Text Corpora - POS tags, named entities, syntactic structures

Access corpora:
```python
import nltk
nltk.corpus.gutenberg.fileids()
```

Find out how many words are in Emma
```python
emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)
```

Browse texts with for loop
```python
for fileid in webtext.fileids():
  print(fileid, webtext.raw(fileid)[:50], '...')

# firefox.txt Cookie Manager: "Don't allow sites that set remove ...
# grail.txt SCENE 1: [wind] [clop clop clop]
# KING ARTHUR: Who ...
# overheard.txt White guy: So, do you have any plans for this even ...
# pirates.txt PIRATES OF THE CARRIBEAN: DEAD MAN'S CHEST, by Ted ...
```

Explore use of modal verbs
```python
from nltk.corpus import brown

news_text = brown.words(categories= 'news')
fdist = nltk.FreqDist([w.lower() for w in news_text])

modals = ['can','could','may','might','must','will']
for m in modals:
   print(m + ':', fdist[m])

 # can: 94, could: 87, may: 93, might: 38, must: 53, will: 389
```

### Text Corpora Structures
![text corpora structures](/img/text_corpora_structures.png)

---

## Conditional Frequency Distributions
A **conditional frequency distribution** is a collection of frequency distributions, each one for a different “condition.” The condition will often be the category of the text.
