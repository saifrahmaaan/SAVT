import os


def load_dataset():
    sentences = []
    labels = []

    data_directory = '../../data'  # Adjust the path according to your project structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, data_directory)

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
