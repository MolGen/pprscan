import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix


def preprocessing_seqs(seqs,
                       alphabet="ACDEFGHIKLMNPQRSTVWY*",
                       maxlen=35,
                       padding="post",
                       truncating="post",
                       flatten=False):
    """
    Sequences encoding schemes
    """
    tokenizer = Tokenizer(num_words=len(alphabet),
                          char_level=True,
                          lower=False)

    tokenizer.fit_on_texts(alphabet)
    sequences = tokenizer.texts_to_sequences(seqs)
    sequences = np.array(pad_sequences(sequences,
                                       maxlen=maxlen,
                                       padding=padding,
                                       truncating=padding))
    encoded_seqs = to_categorical(sequences, len(alphabet))
    if flatten:  # for Multilayer perceptron model
        encoded_seqs = encoded_seqs.reshape(len(seqs), -1)
    return encoded_seqs


def preprocessing_features(features, classes):
    """
    encode given categorical features for each sample.
    """
    encoded_features = label_binarize(features, classes=classes)
    return encoded_features


def compute_class_weight(features, classes):
    """
    Estimate class weights for unbalanced datasets
    class weights will be given by n_samples / (n_classes * np.bincount(y))
    """
    # features = [lable.strip() for lables in features for lable in lables.split(",")]  # flatten
    class_weights = class_weight.compute_class_weight('balanced', 
                                                      classes, 
                                                      features)
    return class_weights


def preprocessing(seqs,
                  features,
                  alphabet,
                  classes,
                  maxlen,
                  flatten=False):
    """
    wrapper for preprocessing seqs and features
    """
    assert len(seqs) == len(features)
    x = preprocessing_seqs(seqs=seqs,
                           alphabet=alphabet,
                           maxlen=maxlen,
                           flatten=flatten)
    y = preprocessing_features(features=features, classes=classes)
    return x, y


def preprocessing_dataset(file,
                          alphabet,
                          classes,
                          maxlen,
                          flatten=False):
    """
    wrapper for preprocessing seqs and features from tsv file
    """
    df = pd.read_csv(file, sep="\t", header=0)
    seqs, features = df.seq, df.feature
    assert len(seqs) == len(features)
    x = preprocessing_seqs(seqs,
                           alphabet=alphabet,
                           maxlen=maxlen,
                           flatten=flatten)
    y = preprocessing_features(features=features, classes=classes)
    return x, y


def train_test_split_print(x, y, test_size=0.2, random_state=23):
    """
    Split arrays or matrices into random train and test subsets
    """
    assert x.shape[0] == y.shape[0]
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    print("x_train.shape:\t{}".format(x_train.shape))
    print("y_train.shape:\t{}".format(y_train.shape))
    print("x_test.shape:\t{}".format(x_test.shape))
    print("y_test.shape:\t{}".format(y_test.shape))
    return x_train, x_test, y_train, y_test


def parse_fasta(fh):
    """
    https://stackoverflow.com/a/7655072
    """
    acc, seq = None, []
    for line in fh:
        line = line.strip()
        if line.startswith(">"):
            if acc:
                yield (acc, ''.join(seq))
            acc, seq = line[1:].split()[0], []
        else:
            seq.append(line)
    if acc:
        yield (acc, ''.join(seq))


def scan_protein(accession, sequence, model, classes, w=35, flatten=False, bg="B"):
    """
    scan a protein for matches against the PPR-like motifs with sliding window size of `w`.
    """
    subsequences = [sequence[i:i + w] for i in range(len(sequence) - w + 1)]
    encoded_subsequences = preprocessing_seqs(subsequences, maxlen=w, flatten=flatten)
    y_prob = model.predict(encoded_subsequences)
    y_classes = y_prob.argmax(axis=-1)

    starts = np.where(classes[y_classes] != bg)[0]
    feature = classes[y_classes][starts]
    proba = y_prob.max(axis=-1)[starts]
    seq = np.array(subsequences)[starts]
    d = {"accession": accession,
         "start": starts,
         "end": starts + w,
         "feature": feature,
         "score": proba,
         "strand": "+",
         "seq": seq}

    df = pd.DataFrame(
        d, columns=['accession', 'start', 'end', 'feature', 'score', 'strand', 'seq'])
    return df


def scan_fasta(fasta, weight, classes, w=35, flatten=False, bg="B"):
    """
    scan proteins for matches against the PPR-like motifs
    """
    model = load_model(weight)
    with open(fasta) as fh:
        entries = pd.DataFrame()
        for acc, seq in parse_fasta(fh):
            entries_temp = scan_protein(
                acc, seq, model, classes=classes, flatten=flatten, bg=bg, w=w)
            entries = entries.append(entries_temp, ignore_index=False)
    return entries


def plot_confusion_matrix(model, x_test, y_test, labels, norm=False, report=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification
    """
    # labels = np.array(labels)
    y_prob = model.predict(x_test)
    y_pred = np.array(labels[y_prob.argmax(axis=1)])
    y_true = np.array(labels[y_test.argmax(axis=1)])

    if report:
        print(classification_report(y_true, y_pred, labels=labels))

    cm = confusion_matrix(y_true, y_pred, labels)
    
    if norm:
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm / np.expand_dims(cm.sum(axis=1), axis=1)
        fmt = ".2g"
    else:
        fmt = 'd'
        
    plt.figure(figsize=(12, 6))
    colors = sns.light_palette((210, 90, 60), input="husl")
    sns.heatmap(cm,
                cmap=colors, #"coolwarm"
                linecolor='white',
                linewidths=0.5,
                xticklabels=labels,
                yticklabels=labels,
                fmt=fmt,
                annot=True)
    
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_history_curves(history):
    """
    Plot model accuracy and loss curves
    """
    epochs = len(history.history['acc'])
    
    plt.subplot(1,2,1)
    plt.tight_layout()
    plt.xticks(range(1, epochs+1, epochs//5))
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.tight_layout()
    plt.xticks(range(1, epochs+1, epochs//5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
