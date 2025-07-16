import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
dataset = pd.read_csv('dataset/restaurant_reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# import re
# import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
all_stopwords = stopwords.words('english')
words_to_remove = ['not', 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'no', 'nor', 'did', 'didn', "didn't", 'can', 'couldn', "couldn't"]
all_stopwords = [word for word in all_stopwords if word not in words_to_remove]
print(all_stopwords)

for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)

  corpus.append(review)


# Creating the Bag of Words model
# from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the Naive Bayes model on the Training set
# from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))