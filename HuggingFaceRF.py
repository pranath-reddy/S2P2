'''
Name: Pranath Reddy Kumbam
UFID: 8512-0977
NLP Project Codebase

Code for loading/processing the UC Berkley "Measuring Hate Speech" dataset from Hugginface and training a Random Forest Model
'''

# Load libraries
import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import datasets

# Load the dataset
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
df = dataset['train'].to_pandas()
df.rename(columns={'text': 'tweet', 'hate_speech_score': 'class'}, inplace=True)
df['binary_class'] = df['class'] > 0.5

# Convert to lowercase, remove punctuation, extra spaces, URLs, mentions, and hashtags
df['tweet'] = df['tweet'].str.lower().replace(r'[^\w\s]', '', regex=True).replace(' {2,}', ' ', regex=True).replace('"', '')
df['tweet'] = df['tweet'].replace(r'http\S+|www.\S+|@\w+|#\w+', '', regex=True)

# Tokenization
nltk.download('punkt')
df['tweet'] = df['tweet'].apply(nltk.word_tokenize)

# Lemmatization
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
df['tweet'] = df['tweet'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Removing stop-words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x if word not in stop_words]))

# Prepare the data using the TF-IDF approach
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(df['tweet']).toarray()
y = df['binary_class']

# Splitting the Data using Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
true_labels, predictions = np.asarray(y_test), np.asarray(y_pred)

# Calculate accuracy, precision, recall, F1-score, and confusion matrix
accuracy = np.mean(np.array(predictions) == np.array(true_labels))
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
conf_mat = confusion_matrix(true_labels, predictions)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1_score)
print("Confusion Matrix:\n", conf_mat)




