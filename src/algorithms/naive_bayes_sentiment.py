import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load and preprocess dataset
def load_dataset(data_dir):
    sentences = []
    labels = []

    with open(os.path.join(data_dir, 'dictionary.txt'), 'r', encoding='utf-8') as dict_file:
        phrases = dict_file.readlines()
        phrase_to_id = {line.split('|')[0]: line.split('|')[1].strip() for line in phrases}

    with open(os.path.join(data_dir, 'sentiment_labels.txt'), 'r', encoding='utf-8') as labels_file:
        label_lines = labels_file.readlines()
        id_to_label = {line.split('|')[0]: float(line.split('|')[1].strip()) for line in label_lines}

    with open(os.path.join(data_dir, 'datasetSentences.txt'), 'r', encoding='utf-8') as sentences_file:
        for line in sentences_file:
            sentence_id, sentence = line.strip().split('\t')
            sentence_id = int(sentence_id)
            if str(sentence_id) in phrase_to_id and phrase_to_id[str(sentence_id)] in id_to_label:
                sentences.append(sentence)
                sentiment_label = id_to_label[phrase_to_id[str(sentence_id)]]
                # Define sentiment classes based on probability cutoffs
                if sentiment_label <= 0.2:
                    labels.append('very negative')
                elif sentiment_label <= 0.4:
                    labels.append('negative')
                elif sentiment_label <= 0.6:
                    labels.append('neutral')
                elif sentiment_label <= 0.8:
                    labels.append('positive')
                else:
                    labels.append('very positive')

    return sentences, labels

# Load and preprocess dataset
data_dir = '../../data'  # Path to the dataset directory from the algorithms directory
current_dir = os.path.dirname(os.path.abspath(__file__))
full_data_dir = os.path.join(current_dir, data_dir)
sentences, labels = load_dataset(full_data_dir)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

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
