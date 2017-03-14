# Helpfulness-Prediction
Prediction of helpfulness of reviews on Amazon using Amazon product review data (https://inclass.kaggle.com/c/cse158-258-helpfulness-prediction)


** Files **

**train.json.gz** 200,000 reviews to be used for training. The fields in this file are:

**itemID**: The ID of the item. This is a hashed product identifier from Amazon.

**reviewerID**: The ID of the reviewer. This is a hashed user identifier from Amazon.

**helpful**: Helpfulness votes for the review. This has two subfields, 'nHelpful' and 'outOf'. The latter is
the total number of votes this review received, the former is the number of those that considered
the review to be helpful.

**reviewText**: The text of the review.

**summary**: Summary of the review.

**price**: Price of the item.

**reviewHash**: Hash of the review (essentially a unique identifier for the review).

**unixReviewTime**: Time of the review in seconds since 1970.

**reviewTime**: Plain-text representation of the review time.

**category**: Category labels of the product being reviewed.

**pairs Helpful.txt** Pairs on which you are to predict helpfulness votes. A third column in this file is the total
number of votes, from which you should predict how many were helpful.

**test_Helpful.json.gz** Review data for the each user-item pair in *pairs Helpful.txt*

Download data files from: https://drive.google.com/open?id=0BzmUFGrGTLhfSFFxNFBaNmxSV1U
