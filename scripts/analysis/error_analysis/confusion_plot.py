print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import codecs

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def main():
  y_test = []
  y_pred = []
  class_names = np.array(['not news', 'news'])

  #filepath = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_text/BERT_bert_text_test_results.txt'
  filepath = '/Users/aggarwalpiush/github_repos/tweetnewsdetection/test_results_newsnotnews_text_newsnotnews/SVC_emb_text_newsnotnews_test_results.txt'

  with codecs.open(filepath, 'r', 'utf-8') as prediction_obj:
    for line in prediction_obj:
      tokens = line.split('\t')
      y_pred.append(int(tokens[0].strip().replace('\n','').rstrip('\r\n')))
      y_test.append(int(tokens[1].strip().replace('\n','').rstrip('\r\n')))

  y_test = np.array(y_test)
  y_pred = np.array(y_pred)
  np.set_printoptions(precision=2)

  # Plot non-normalized confusion matrix
  #plot_confusion_matrix(y_test, y_pred, classes=class_names,
   #                     title='Confusion matrix, without normalization')

  # Plot normalized confusion matrix
  plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)

  plt.show()


if __name__ == '__main__':
  main()
