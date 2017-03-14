"""
Created on Sat Feb  4 18:21:31 2017

@author: rmisra
"""
import gzip
from collections import defaultdict
import numpy
import time
from time import strptime
import random
import string
from sklearn.preprocessing import PolynomialFeatures
from nltk.stem.porter import PorterStemmer

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


#### For random shuffling of data
seed = 3
random.seed(seed)
shuffle = [i for i in range(0,200000)]
random.shuffle(shuffle)

#### Variable initialization
training_size = 200000

first_model_train_label = []
second_model_train_label = []

X_train_first_model = numpy.ones((training_size, 6))
X_train_second_model = numpy.ones((training_size, 6))

counter = 0      
val_index = 0  
train_index_first_model = 0    
train_index_second_model = 0  

punctuation = set(string.punctuation)
stemmer = PorterStemmer()


#### Retrieved Preprocessed available price for items
item_price = {}
file = open("Item_price.txt", 'r')
for l in file.readlines():
    line = l.strip().split(',')
    item_price[line[0]] = float(line[1]) 

file.close()

#### If price of item not present in data, replace it with pre-calculated mean category price
mean_price_per_cat = {0:22.899447041065883, 1:32.40620099196726, 2:11.99711672473867, 3:14.564745222929853, 4:13.220347666971549}



#### Preparing feature matrices for training and validation
for l in readGz("train.json.gz"):

    counter += 1
    index = -1

    ## randomly deciding train and validation samples
    if(shuffle[counter-1] < training_size):
        
        ## Preparing feature matrix for first model
        if l['helpful']['outOf'] > 16 and l['helpful']['outOf'] < 200:
            
            first_model_train_label.append(l['helpful']['nHelpful']/l['helpful']['outOf']);

            # basic text processing
            r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation])
            words = r.split(' ')

            text = defaultdict(int)
            for w in words:
                w = stemmer.stem(w)
                text[w] += 1

            # feature 1 -> number of words after pre-processing
            index +=1; X_train_first_model[train_index_first_model,index] = (len(text))

            # feature 2 -> rating given by user to the item
            index +=1; X_train_first_model[train_index_first_model,index] = (l['rating'])

            # feature 3 -> price of the item
            index +=1; X_train_first_model[train_index_first_model,index] = item_price[l['itemID']] if l['itemID'] in item_price else mean_price_per_cat[l['categoryID']]
            
            # feature 4 -> month in number (1-12)
            index +=1; X_train_first_model[train_index_first_model,index] = strptime(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(l['unixReviewTime'])).split(' ')[2],'%b').tm_mon
                        
            # feature 5 -> category id
            index +=1; X_train_first_model[train_index_first_model,index] = l['categoryID']
            
            # feature 6 -> total votes for the review
            index +=1; X_train_first_model[train_index_first_model,index] = l['helpful']['outOf']            

            
            train_index_first_model += 1
        
        
        ## Preparing feature matrix for second model -> feature configuration is same as above    
        elif l['helpful']['outOf'] > 5 and l['helpful']['outOf'] <= 16:
            
            second_model_train_label.append(l['helpful']['nHelpful']/l['helpful']['outOf']);

            r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation])
            words = r.split(' ')

            text = defaultdict(int)
            for w in words:
                w = stemmer.stem(w)
                text[w] += 1
            
            
            index+=1; X_train_second_model[train_index_second_model,index] = (len(text))

            index+=1; X_train_second_model[train_index_second_model,index] = (l['rating'])

            index+=1; X_train_second_model[train_index_second_model,index] = item_price[l['itemID']] if l['itemID'] in item_price else mean_price_per_cat[l['categoryID']]
            
            index+=1; X_train_second_model[train_index_second_model,index]  = strptime(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(l['unixReviewTime'])).split(' ')[2],'%b').tm_mon
            
            index+=1; X_train_second_model[train_index_second_model,index]  = l['categoryID']
            
            index+=1; X_train_second_model[train_index_second_model,index] = l['helpful']['outOf']
                                    
            train_index_second_model += 1



#### Reducing the dimensions based on the data stored
X_train_first_model = X_train_first_model[:train_index_first_model,:]
X_train_second_model = X_train_second_model[:train_index_second_model,:]

print("feature matrices prepared")


#### polynomial feature preprocessing; to consider interactions between features
poly = PolynomialFeatures(2, interaction_only=True, include_bias = True)
poly2 = PolynomialFeatures(2, interaction_only=True, include_bias = True)

poly.fit(X_train_first_model)
poly2.fit(X_train_second_model)

X_train_first_model = poly.transform(X_train_first_model)
X_train_second_model = poly2.transform(X_train_second_model)


#### Linear Regression models
theta1,_,_,_ = numpy.linalg.lstsq(X_train_first_model, first_model_train_label)

theta2,_,_,_ = numpy.linalg.lstsq(X_train_second_model, second_model_train_label)


## Prepare test features
X_test = []
for l in readGz("test_Helpful.json.gz"):
    if(l['helpful']['outOf'] != 0):
        
        feature = []

        r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation])
        words = r.split(' ')
        text = defaultdict(int)
        for w in words:
            w = stemmer.stem(w)
            text[w] += 1
            
        
        feature.append(len(text))
        
        feature.append((l['rating']))

        feature.append(item_price[l['itemID']] if l['itemID'] in item_price else mean_price_per_cat[l['categoryID']])
        
        feature.append(strptime(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(l['unixReviewTime'])).split(' ')[2],'%b').tm_mon)
        
        feature.append(l['categoryID'])
        
        feature.append(l['helpful']['outOf'])

        X_test.append(feature)

## make predictions
prediction_test = []
for i in range(len(X_test)):
    ## use the first model if outOf is greater than 16s else use the second model
    if X_test[i][5] > 16:
        prediction_test.append(numpy.dot(poly.transform(numpy.array(X_test[i]).reshape(1,-1)), theta1)[0])
    else:
        prediction_test.append(numpy.dot(poly2.transform(numpy.array(X_test[i]).reshape(1,-1)), theta2)[0])  


## save test predictions to a file
predictions = open("./predictions_Helpful.txt", 'w')

for l in open("pairs_Helpful.txt"):
    if l.startswith("userID"):
        predictions.write(l)
        break

i=0;
for l in readGz("test_Helpful.json.gz"):
    ## Predict 0 wherever outOf is 0 else write the prediction we made into the file
    if(l['helpful']['outOf'] == 0):
        predictions.write(l['reviewerID'] + '-' + l['itemID'] + '-' + str(0) + ',' + str(0) + '\n')
    else:
        predictions.write(l['reviewerID'] + '-' + l['itemID'] + '-' + str(l['helpful']['outOf']) + ',' + str(round(prediction_test[i]*l['helpful']['outOf'])) + '\n')
        i+=1

predictions.close()

print('predictions generated')