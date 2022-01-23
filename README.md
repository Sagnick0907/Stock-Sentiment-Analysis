# Stock-Sentiment-Analysis

# SpamClassifier

## Table of Content
  * [Overview](#overview)
  * [Technical Aspect](#technical-aspect)
  * [Technologies Used](#technologies-used)

## Overview
This is a Stock Sentiment Analyzer trained with Random Forest classifier. 
![image](https://user-images.githubusercontent.com/76872499/150687106-f8d4a061-cd88-434a-acf6-d23900ce533b.png)

## Technical Aspect
Given below are the steps taken to build the model:  
For the NLP implementation code –  
  - Imported necessary **Python libraries** (Pandas, re, nltk, sklearn, etc.)
  -	Imported the Dataset and performed a **train_test_split** on labels & messages.
  -	In **Data cleaning** & **Text pre-processing** Section, we removed all unnecessary symbols & numbers. We lower cased all sentences & split sentences into words.
  -	Performed **removal of stopwords**.
  -	Created the **Bag of Words model** and where for **CountVectorizer** we select ngram_range=(2,2)(all combinations of 2 words).
  -	We **transformed** both Training dataset and Testing dataset using **CountVectorizer**.
  -	Trained & tested the model using **RandomForestClassifier**.
  -	From sklearn we imported **metrics** to produce classification_report,confusion_matrix & confusion_matrix for our model.  

## Technologies Used
- Spyder IDE
-	ML model: Random Forest Classifier (Selected)
-	Libraries: pandas, re, nltk, sk-learn.
