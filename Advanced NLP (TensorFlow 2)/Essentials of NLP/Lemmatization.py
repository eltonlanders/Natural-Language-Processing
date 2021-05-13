# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:36:45 2021

@author: elton
"""

import stanza
en = stanza.download('en') 
en = stanza.Pipeline(lang='en')



# Test text for Stemming and Lemmatization
text = '''
Sherlock Holmes laughed. “Here is the foresight,” said he putting his
finger upon the little disc and loop of the hat-securer. “They are
never sold upon hats. If this man ordered one, it is a sign of a
certain amount of foresight, since he went out of his way to take this
precaution against the wind. But since we see that he has broken the
elastic and has not troubled to replace it, it is obvious that he has
less foresight now than formerly, which is a distinct proof of a
weakening nature. On the other hand, he has endeavoured to conceal some
of these stains upon the felt by daubing them with ink, which is a sign
that he has not entirely lost his self-respect.”'''

lemma = en(text)

# Printing the Lemmas along with the POS of each word
lemmas = ""
for sentence in lemma.sentences:
        for token in sentence.tokens:
            lemmas += token.words[0].lemma + "/" + token.words[0].upos + " "
        lemmas += "\n"
print(lemmas)
