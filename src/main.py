from algorithms.naive_bayes_sentiment import train_and_test_naive_bayes
from preprocessing.pre_processing import load_dataset
from visualisation.visualise_data import visualize_distribution, visualize_sentence_length, create_word_cloud
from sklearn.model_selection import train_test_split

def main():
    sentences, labels = load_dataset()

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    train_and_test_naive_bayes(X_train, y_train, X_test, y_test)

    # Create a sample dataframe for visualization
    import pandas as pd
    data = {'sentiment': labels}  # Assuming labels contain sentiment categories
    dataframe = pd.DataFrame(data)

    # Sample positive text for word cloud
    positive_text = "This is a sample positive text for visualization purposes."
    # Sample list of sentence lengths for visualization
    sentence_lengths = [len(sentence.split()) for sentence in sentences]

    # Call visualization functions
    visualize_distribution(dataframe)  # Call this function with the appropriate DataFrame
    create_word_cloud('positive', positive_text)  # Call this function with the appropriate sentiment and text
    visualize_sentence_length(sentence_lengths)  # Call this function with the list of sentence lengths

if __name__ == "__main__":
    main()
