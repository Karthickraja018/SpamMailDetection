import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
data = pd.read_csv('/content/spam_ham_dataset.csv')
X = data['text']
y = data['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
model = SVC(probability=True)
model.fit(X_train_tfidf, y_train)

def predict_email(email_text):
    email_tfidf = tfidf.transform([email_text])
    prob = model.predict_proba(email_tfidf)[0, 1]
    if prob > 0.5:
        classification = "Spam"
    else:
        classification = "Ham"
    
    return classification, prob
while True:
    print("\nEmail Spam Classifier")
    print("Enter 'quit' to exit")
    email_text = input("Enter the email text to classify: ")
    if email_text.lower() == 'quit':
        break
    result, probability = predict_email(email_text)
    print(f"\nClassification: {result}")
    print(f"Probability of being spam: {probability:.2f}")
