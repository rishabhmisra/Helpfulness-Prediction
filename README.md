# Helpfulness-Prediction
Prediction of helpfulness of reviews on Amazon using Amazon product review data

Files
train.json.gz 200,000 reviews to be used for training. It is not necessary to use all reviews for training, for
example if doing so proves too computationally intensive. While these files are one-json-per-line (much as
we have seen so far in class), you may find it useful to represent them more concisely in order to produce
a more efficient solution. The fields in this file are:

itemID: The ID of the item. This is a hashed product identifier from Amazon.
reviewerID: The ID of the reviewer. This is a hashed user identifier from Amazon.
helpful: Helpfulness votes for the review. This has two subfields, `nHelpful' and `outOf'. The latter is
the total number of votes this review received, the former is the number of those that considered
the review to be helpful.
reviewText: The text of the review. It should be possible to successfully complete this assignment
without making use of the review data, though an eective solution to the helpfulness prediction
task will presumably make use of it.
summary Summary of the review.
price Price of the item.
reviewHash Hash of the review (essentially a unique identier for the review).
unixReviewTime Time of the review in seconds since 1970.
reviewTime Plain-text representation of the review time.
category Category labels of the product being reviewed.
pairs Helpful.txt Pairs on which you are to predict helpfulness votes. A third column in this le is the total
number of votes, from which you should predict how many were helpful.
pairs Category.txt Pairs (userID and reviewHash) on which you are to predict the category of an item. Not
relevant for CSE258
pairs Rating.txt Pairs (userIDs and itemIDs) on which you are to predict ratings.
helpful.json.gz The review data associated with the helpfulness prediction test set. The `nHelpful' eld has
been removed from this data, since that is the value you need to predict above. This data will only be
of use for the helpfulness prediction task.
category.json.gz The review data associated with the category prediction test set. Again, the eld that you
are trying to predict has been removed.
baselines.py A simple baseline for each task, described below.
Please do not try to crawl these products from Amazon, or to reverse-engineer the hashing function I
used to anonymize the data. I assure you that doing so will not be easier than successfully completing the
assignment.
