# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:08:59 2021

@author: elton
"""
from bs4 import BeautifulSoup
import pandas as pd
import requests



# Scraping a test site
titles = []
prices = []
ratings = []
url = 'https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops'
request = requests.get(url)
soup = BeautifulSoup(request.text, "html.parser")
for product in soup.find_all('div', {'class': 'col-sm-4 col-lg-4 col-md-4'}):
    for pr in product.find_all('div', {'class': 'caption'}):
        for p in pr.find_all('h4', {'class': 'pull-right price'}):
            prices.append(p.text)
        for title in pr.find_all('a' , {'class': 'title'}):
            titles.append(title.get('title'))
    for rt in product.find_all('div', {'class': 'ratings'}):
        ratings.append(len(rt.find_all('span', 
                                       {'class': 'glyphicon glyphicon-star'})))


# Building a dataframe and exporting to a csv file            
product_df = pd.DataFrame(zip(titles,prices,ratings), columns =['Titles', 'Prices', 'Ratings'])  
product_df.to_csv("ecommerce.csv",index=False)