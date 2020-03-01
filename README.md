# Homework 2: Text Classification with Naive Bayes and Logistic Regression

## Description
This homework will expose you to scikit-learn: a Python API that is used for common NLP
and Machine Learning tasks. Specifically, you will learn how to use scikit-learn to carry out
feature engineering and supervised learning for sentiment classification of movie reviews.

Download and unzip the training and test corpora available on the class webpage.
Datasets are simple plaintext files grouped into two folders: pos and neg. All files in
the pos folder have a positive sentiment associated with them; and all files in the neg
folder have a negative sentiment associated with them.
- Use the CountVectorizer and TfidfVectorizer classes provided by scikit-learn to obtain
bag-of-words and tf-idf representations of the raw text respectively.
- With the feature representation as input; train the Naive Bayes and Logistic Regression
classifier(s) to carry out text classification.
- Test the performance of your classifier(s) on the test set by reporting accuracy, precision, recall and F-score values for the test set.
Additionally, carry out these experiments:
- Observe the effect of using bag-of-words and tf-idf representations on the model’s
performance.
- Look into how stop words can be removed. Observe the effect of removing stop words
on model performance.
- Observe the effect of L1 and L2 regularization v/s no regularization with Logistic
Regression on model performance.

## Instructions

1. Unzip the aclImdb_v1.tar.gz file.

2. Install dependencies:
    _Linux or macOS_
    ```bash
    pip3 install -r requirements.txt
    ```

    _Windows_
    ```bash
    pip install -r requirements.txt
    ```

3. To run, type in the command line interpreter:

    _Linux or macOS_
    ```bash
    python3 hw2.py <path-to-train-set> <path-to-test-set> <representation> <classifier> <stop-words> <regularization>
    ```

    _Windows_
    ```bash
    python hw2.py <path-to-train-set> <path-to-test-set> <representation> <classifier> <stop-words> <regularization>
    ```

    #### Valid arguments:
    - ```representation ∈ {bow, tfidf}```
    - ```classifier ∈ {nbayes, regression}```
    - ```stop-words ∈ {0, 1}```
    - ```regularization ∈ {no, l1, l2}```

*NOTE*: Python version >=3.6.1 is recommended.
