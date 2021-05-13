# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:26:22 2021

@author: elton
"""
import stanza
en = stanza.download('en') 
en = stanza.Pipeline(lang='en')



# Test text
text = ''' I had called upon my friend, Mr. Sherlock Holmes, one day in the
autumn of last year and found him in deep conversation with a very
stout, florid-faced, elderly gentleman with fiery red hair. With an
apology for my intrusion, I was about to withdraw when Holmes pulled
me abruptly into the room and closed the door behind me.'''

# Processing the text
pos = en(text)

# Printing back the sentence tokens with the POS tags
def print_pos(doc):
    text = ""
    for sentence in doc.sentences:
        for token in sentence.tokens:
            text += token.words[0].text + "/" + token.words[0].upos + " "
        text += "\n"
    return text
print(print_pos(pos))


# Test text 2
text_2 = '''
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

pos = en(text_2)

def print_pos(doc):
    text = ""
    for sentence in doc.sentences:
        for token in sentence.tokens:
            text += token.words[0].text + "/" + token.words[0].upos + " "
        text += "\n"
    return text
print(print_pos(pos))