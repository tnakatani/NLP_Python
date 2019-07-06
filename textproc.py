# Calculate lexical diversity
from __future__ import division
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

wish = plural('wish')
hope = plural('hope')
print(wish)
print(hope)
