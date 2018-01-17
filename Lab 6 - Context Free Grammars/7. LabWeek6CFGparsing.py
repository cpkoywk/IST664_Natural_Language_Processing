
# coding: utf-8

# In[1]:

# Context Free Grammars and Parsing in the NLTK
import nltk


# In[2]:

nltk.app.srparser()


# In[3]:

## write your own grammars
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" 
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)


# In[4]:

# top-down method: recursive descent parsing
# create the parser from a grammar
rd_parser = nltk.RecursiveDescentParser(grammar)


# In[6]:

# make an example sentence text
senttext = "Mary saw Bob"
# tokenize the sentence by splitting on white space
# use nltk.word_tokenize() for more complex examples
sentlist = senttext.split()
print(sentlist)


# In[7]:

# run the parse function on the list of tokens
trees = rd_parser.parse(sentlist)
# convert the generator to a list
treelist = list(trees)
# what type is an individual tree?
print(type(treelist[0]))
# print the tree structures
for tree in treelist:
	print (tree)


# In[8]:

# try an ambiguous sentence
sent2list = "John saw the man in the park with a telescope".split()
for tree in rd_parser.parse(sent2list):
	print (tree)


# In[9]:

# extend the grammar with more words (I, elephant, pajamas)
groucho_grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked" | "shot"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" | "I"
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "cat" | "telescope" | "park" | "elephant" | "pajamas"
  P -> "in" | "on" | "by" | "with"
  """)


# In[10]:

# if we change the grammar, we create a new parser
rd_parser = nltk.RecursiveDescentParser(groucho_grammar)

# try sent4 with the recursive descent parser on groucho grammar
sent4list = "I shot an elephant in my pajamas".split()
for tree in rd_parser.parse(sent4list):
	print (tree)


# In[11]:

# extend the grammar for the flight grammar:  adding a rule to S and some words
flight_grammar = nltk.CFG.fromstring("""
  S -> NP VP | VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked" | "shot" | "book"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" | "I"
  Det -> "a" | "an" | "the" | "my" | "that"
  N -> "man" | "dog" | "cat" | "telescope" | "park" | "elephant" | "pajamas" | "flight"
  P -> "in" | "on" | "by" | "with"
  """)


# In[12]:

# make a recursive descent parser and parse the sentence
rd_parser = nltk.RecursiveDescentParser(flight_grammar)
sent5list = 'book that flight'.split()
for tree in rd_parser.parse(sent5list):
	print (tree)


# In[14]:

## Look at Dependency grammars in the NLTK book, section 8.5
# a dependency grammar for the groucho example
# note difficulty of writing rules for every word dependency
groucho_dep_grammar = nltk.DependencyGrammar.fromstring("""
  'shot' -> 'I' | 'elephant' | 'in'
  'elephant' -> 'an' | 'in'
  'in' -> 'pajamas'
  'pajamas' -> 'my'
  """)
print (groucho_dep_grammar)


# In[15]:

# create a dependency parser, assumes sentence is projective
pdp = nltk.ProjectiveDependencyParser(groucho_dep_grammar)
glist = 'I shot an elephant in my pajamas'.split()
# use the parse function to parse a list of tokens
trees = pdp.parse(glist)
for tree in trees:
    print (tree)


# In[16]:

## Probabilistic CFG with verb subcategories
#    for transitive (TranV), intransitive (InV) and dative (DatV) verbs
prob_grammar = nltk.PCFG.fromstring("""
  S -> NP VP [0.9]| VP  [0.1]
  VP -> TranV NP [0.3]
  VP -> InV  [0.3]
  VP -> DatV NP PP  [0.4]
  PP -> P NP   [1.0]
  TranV -> "saw" [0.2] | "ate" [0.2] | "walked" [0.2] | "shot" [0.2] | "book" [0.2]
  InV -> "ate" [0.5] | "walked" [0.5]
  DatV -> "gave" [0.2] | "ate" [0.2] | "saw" [0.2] | "walked" [0.2] | "shot" [0.2]
  NP -> Prop [0.2]| Det N [0.4] | Det N PP [0.4]
  Prop -> "John" [0.25]| "Mary" [0.25] | "Bob" [0.25] | "I" [0.25] 
  Det -> "a" [0.2] | "an" [0.2] | "the" [0.2] | "my" [0.2] | "that" [0.2]
  N -> "man" [0.15] | "dog" [0.15] | "cat" [0.15] | "park" [0.15] | "telescope" [0.1] | "flight" [0.1] | "elephant" [0.1] | "pajamas" [0.1]
  P -> "in" [0.2] | "on" [0.2] | "by" [0.2] | "with" [0.2] | "through" [0.2]
  """)


# In[17]:

# create a viterbi parser, a parser that expects a PCFG
viterbi_parser = nltk.ViterbiParser(prob_grammar)
# use its parse function on a list of tokens
for tree in viterbi_parser.parse(['John', 'saw', 'a', 'telescope']):
    print (tree)


# In[18]:

# parse some other sentences
# this parser chooses to return the highest probability tree
for tree in viterbi_parser.parse(sent2list):
    print (tree)


# In[19]:

for tree in viterbi_parser.parse(sent4list):
    print (tree)


# In[20]:

## Exercise
# Define sentences for the exercise
sentex1 = "I prefer a flight through Houston".split()
sentex2 = "Jack walked with the dog".split()
sentex3 = "John gave the dog a bone".split()
sentex4 = "I want to book that flight".split()


# In[21]:

# extend the flight grammar:
flight_grammar = nltk.CFG.fromstring("""
  S -> NP VP | VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked" | "shot" | "book"
  NP -> Prop | Det N | Det N PP
  Prop -> "John" | "Mary" | "Bob" | "I"
  Det -> "a" | "an" | "the" | "my" | "that"
  N -> "man" | "dog" | "cat" | "telescope" | "park" | "elephant" | "pajamas" | "flight"
  P -> "in" | "on" | "by" | "with"
  """)


# In[22]:

# redefine rd_parser when you change the flight grammar
rd_parser = nltk.RecursiveDescentParser(flight_grammar)
for tree in rd_parser.parse(sentex1):   print (tree)


# In[ ]:



