import pandas as pd
df=pd.read_csv('C:/Users/91943/PROJECT/Stock Price Sentiment Analysis/Data.csv', encoding = "ISO-8859-1")

X=df.iloc[:,2:27]
y=df['Label']

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
X.columns= new_Index

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


headlines = []
for row in range(0,len(X_train.index)):
    headlines.append(' '.join(str(x) for x in X_train.iloc[row,0:25]))

import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

corpus = []
for i in range(len(headlines)):
    text = re.sub('[^a-zA-Z]', ' ', headlines[i])
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
    print(i," Completed")

# Implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
train_f=countvector.fit_transform(corpus)


# Implement RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(train_f,y_train)
print("Training Completed Successfully!")


# Predict for the Test Dataset
test_transform = []
for row in range(0,len(X_test.index)):
    test_transform.append(' '.join(str(x) for x in X_test.iloc[row,0:25]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(y_test,predictions)
print(matrix)
score=accuracy_score(y_test,predictions)
print(score)
report=classification_report(y_test,predictions)
print(report)
