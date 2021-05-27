#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 20:02:47 2021

@author: elton
"""
#!pip install wikipedia==1.4.0
#!pip install ktrain==0.26.2
import wikipedia
from ktrain import text



# Specifying the title of the Wikipedia page which to be extracted
wiki = wikipedia.page('Pablo Picasso')

# Extracting the plain text content of the page
doc = wiki.content

# Printing first 1000 words
print(doc[:1000])
"""
Pablo Ruiz Picasso (25 October 1881 – 8 April 1973) was a Spanish painter, 
sculptor, printmaker, ceramicist and theatre designer who spent most of his 
adult life in France. Regarded as one of the most influential artists of the 
20th century, he is known for co-founding the Cubist movement, the invention 
of constructed sculpture, the co-invention of collage, and for the wide variety
of styles that he helped develop and explore. Among his most famous works are
the proto-Cubist Les Demoiselles d'Avignon (1907), and Guernica (1937), a 
dramatic portrayal of the bombing of Guernica by German and Italian air forces
during the Spanish Civil War. Picasso demonstrated extraordinary artistic 
talent in his early years, painting in a naturalistic manner through his 
childhood and adolescence. During the first decade of the 20th century, 
his style changed as he experimented with different theories, techniques, 
and ideas. After 1906, the Fauvist work of the slightly older artist Henri 
Matisse motivate

"""

# Instantiating the model
ts = text.TransformerSummarizer()

# Summarizing
ts.summarize(doc)
"""
"Pablo Ruiz Picasso (25 October 1881 – 8 April 1973) was a Spanish painter, 
sculptor, printmaker, ceramicist and theatre designer. He is known for 
co-founding the Cubist movement, the invention of constructed sculpture, 
the co-invention of collage, and for the wide variety of styles that he helped 
develop and explore. Among his most famous works are the proto-Cubist Les 
Demoiselles d'Avignon (1907), and Guernica (1937)"
"""



# Testing Netflix wiki page
wiki = wikipedia.page('Netflix')

doc = wiki.content

print(doc[:1500])
"""
Netflix, Inc. is an American over-the-top content platform and production 
company headquartered in Los Gatos, California. Netflix was founded in 1997 by
Reed Hastings and Marc Randolph in Scotts Valley, California. The company's 
primary business is a subscription-based streaming service offering online 
streaming from a library of films and television series, including those 
produced in-house. In April 2021, Netflix had 208 million subscribers, 
including 74 million in the United States and Canada. It is available 
worldwide except in mainland China (due to local restrictions), and Syria, 
North Korea, and Crimea (due to US sanctions). In 2020,  Netflix's operating 
ncome was $1.2 billion. The company has offices in Canada, France, Brazil, 
the Netherlands, India, Japan, South Korea, and the United Kingdom. Netflix 
is a member of the Motion Picture Association (MPA), producing and 
distributing content from countries all over the globe.
Netflix's initial business model included DVD sales and rental by mail, 
but Hastings abandoned the sales about a year after the company's founding to 
focus on the initial DVD rental business. Netflix expanded its business in 
2007 with the introduction of streaming media while retaining the DVD and 
Blu-ray rental business. The company expanded internationally in 2010 with 
streaming available in Canada, followed by Latin America and the Caribbean. 
Netflix entered the content-production industry in 2013, debuting its first 
series House of Cards.
Since 

"""

ts.summarize(doc)
"""
"Netflix was founded in 1997 by Reed Hastings and Marc Randolph. The company's 
primary business is a subscription-based streaming service. In April 2021, 
Netflix had 208 million subscribers, including 74 million in the United States 
and Canada. Netflix released an estimated 126 original series and films in 2016,
 more than any other network or cable channel."
"""



# Testing Amazon wiki page
wiki = wikipedia.page('Amazon (company)')

doc = wiki.content

print(doc[:1500])
"""
Amazon.com, Inc. ( AM-ə-zon) is an American multinational technology company 
based in Seattle, Washington, which focuses on e-commerce, cloud computing, 
digital streaming, and artificial intelligence. It is one of the Big Five 
companies in the U.S. information technology industry, along with Google, 
Apple, Microsoft, and Facebook. The company has been referred to as "one of 
the most influential economic and cultural forces in the world", as well as 
the world's most valuable brand.Jeff Bezos founded Amazon from his garage in 
Bellevue, Washington, on July 5, 1994. It started as an online marketplace for
books but expanded to sell electronics, software, video games, apparel, 
furniture, food, toys, and jewelry. In 2015, Amazon surpassed Walmart as the 
most valuable retailer in the United States by market capitalization. 
In 2017, Amazon acquired Whole Foods Market for US$13.4 billion, which 
ubstantially increased its footprint as a physical retailer. In 2018, its 
two-day delivery service, Amazon Prime, surpassed 100 million subscribers 
worldwide.Amazon is known for its disruption of well-established industries 
through technological innovation and mass scale. It is the world's largest 
nline marketplace, AI assistant provider, live-streaming platform and cloud 
computing platform as measured by revenue and market capitalization. Amazon 
is the largest Internet company by revenue in the world. It is the second 
largest private employer in the United States and one of the world's most 

"""

ts.summarize(doc)
"""
'Amazon.com, Inc. is an American multinational technology company based in 
Seattle, Washington. It focuses on e-commerce, cloud computing, digital 
streaming, and artificial intelligence. In 2015, Amazon surpassed Walmart as 
the most valuable retailer in the United States by market capitalization. 
In 2017, Amazon acquired Whole Foods Market for US$13.4 billion.'
"""
