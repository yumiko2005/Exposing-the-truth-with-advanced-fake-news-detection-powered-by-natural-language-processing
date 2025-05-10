import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load your fake news dataset
df = pd.read_csv('fake_news.csv')  # Must have 'text' and 'label' columns
df = df[['text', 'label']]
df = df.dropna()

# Preprocess and split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved successfully.")
