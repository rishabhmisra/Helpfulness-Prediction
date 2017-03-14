# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 18:21:31 2017

@author: rmisra
"""
import gzip

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


## mean price per category

category_p = {}
category_s = {}

for l in readGz("train.json.gz"):
    if l['categoryID'] not in category_p and 'price' in l:
        category_p[l['categoryID']] = l['price']
        category_s[l['categoryID']] = 1
    elif l['categoryID'] in category_p and 'price' in l:
        category_p[l['categoryID']] += l['price']
        category_s[l['categoryID']] += 1
                  
            
for key in category_p:
    print(str(key) + ' ->' + str(category_p[key]/category_s[key]))


## prepair price file
dict_i = {}

training_size = 200000
file = open("Item_price.txt", 'w')
counter = 0
for l in readGz("train.json.gz"):
    counter+=1
    if (counter<=training_size):
        if(l['itemID'] not in dict_i and 'price' in l):
            dict_i[l['itemID']] = l['price']
    else:
        break
            
for key in dict_i:
    file.write(str(key) + ',' + str(dict_i[key]) + '\n')
        
file.close()