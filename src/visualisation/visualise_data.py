import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def visualize_distribution(dataframe):
    sns.countplot(x='sentiment', data=dataframe)
    plt.xlabel('Sentiment Labels')
    plt.ylabel('Count')
    plt.title('Distribution of Sentiment Labels')
    plt.show()

def create_word_cloud(sentiment, text):
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.show()

def visualize_sentence_length(sentence_lengths):
    sns.histplot(sentence_lengths, bins=20, kde=True)
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentence Lengths')
    plt.show()
