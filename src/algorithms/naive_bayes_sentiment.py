from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def train_and_test_naive_bayes(X_train, y_train, X_test, y_test):
    # Create bag-of-words representation
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    # Train Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_bow, y_train)

    # Test the model
    y_pred = nb_model.predict(X_test_bow)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")