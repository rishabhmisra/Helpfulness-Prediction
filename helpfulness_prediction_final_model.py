"""
Created on Sat Feb  4 18:21:31 2017

@author: rmisra
"""

#### IMPORT LIBRARIES
import gzip
from collections import defaultdict
import numpy
import time
from time import strptime
import random
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import PolynomialFeatures


#### Read .gz file
def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

#### For random shuffling of data
seed = 3
random.seed(seed)
shuffle = [i for i in range(0,200000)]
random.shuffle(shuffle)

#### Variable initialization
training_size = 170000
validation_size = 30000
first_model_train_label = []
second_model_train_label = []
val_label_nHelpful = []
val_label_outOf = []

X_train_first_model = numpy.ones((training_size, 6))
X_train_second_model = numpy.ones((training_size, 6))
X_val = numpy.ones((validation_size, 6))

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
        
    ## validation samples                       
    else:
        ## taking only the samples where total votes are non-zero; would be predicting zeros otherwise
        if l['helpful']['outOf'] != 0:

            val_label_nHelpful.append(l['helpful']['nHelpful'])
            val_label_outOf.append(l['helpful']['outOf'])
            
            
            r = ''.join([c for c in l['reviewText'].lower() if not c in punctuation])
            words = r.split(' ')
            text = defaultdict(int)
            for w in words:
                w = stemmer.stem(w)
                text[w] += 1

            
            index +=1; X_val[val_index,index] = (len(text))

            index +=1; X_val[val_index,index] = (l['rating'])
            
            index +=1; X_val[val_index,index] = item_price[l['itemID']] if l['itemID'] in item_price else mean_price_per_cat[l['categoryID']]
            
            index +=1; X_val[val_index,index] = strptime(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(l['unixReviewTime'])).split(' ')[2],'%b').tm_mon

            index +=1; X_val[val_index,index] = l['categoryID']            

            index +=1; X_val[val_index,index] = l['helpful']['outOf']
            

            val_index += 1



#### Reducing the dimensions based on the data stored
X_train_first_model = X_train_first_model[:train_index_first_model,:]
X_train_second_model = X_train_second_model[:train_index_second_model,:]
X_val = X_val[:val_index,:]

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



#### Calculating error on validation
predictions = []

for i in range(X_val.shape[0]):
    ## if outOf less than or equal to 16, use second model to predict else first
    if X_val[i][5] <= 16:
        predictions.append(numpy.dot(poly2.transform(X_val[i].reshape(1,-1)), theta2)[0])
    else:
        predictions.append(numpy.dot(poly.transform(X_val[i].reshape(1,-1)), theta1)[0])

## calculating Mean Absolute Error
abs_error = sum([abs(val_label_nHelpful[i]-round(predictions[i]*val_label_outOf[i])) for i in range(0,len(val_label_nHelpful))])
print("Mean Absolute Error = " + str(abs_error/validation_size))