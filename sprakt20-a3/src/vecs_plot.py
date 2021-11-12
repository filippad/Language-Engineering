import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from word2vec.w2v import Word2Vec
from RandomIndexing.random_indexing import RandomIndexing
import numpy as np


def draw_interactive(x, y, text):
    """
    Draw a plot visualizing word vectors with the posibility to hover over a datapoint and see
    a word associating with it
    
    :param      x:     A list of values for the x-axis
    :type       x:     list
    :param      y:     A list of values for the y-axis
    :type       y:     list
    :param      text:  A list of textual values associated with each (x, y) datapoint
    :type       text:  list
    """
    norm = plt.Normalize(1,4)
    fig,ax = plt.subplots()
    sc = plt.scatter(x, y, c='b', s=100, cmap=plt.cm.RdYlGn, norm=norm)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        note = "{}".format(" ".join([text[n] for n in ind["ind"]]))
        annot.set_text(note)
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def load(fname):
    """
    Load the word2vec model from a file `fname`
    """
    with open(fname, 'r') as f:
        V, H = (int(a) for a in next(f).split())
        W, words= np.zeros((V, H)), []
        for i, line in enumerate(f):
            parts = line.split()
            word = parts[0].strip()
            words.append(word)
            W[i] = list(map(float, parts[1:]))

    return words, W


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='embedding visualization toolkit')
    parser.add_argument('file', type=str, help='A textual file containing word vectors')
    parser.add_argument('-v', '--vector-type', default='w2v', choices=['w2v', 'ri'])
    parser.add_argument('-d', '--decomposition', default='svd', choices=['svd', 'pca'],
                        help='Your favorite decomposition method')
    args = parser.parse_args()

    words, W = load(args.file)
    svd = TruncatedSVD()
    svd.fit(W)
    reduced_ver = svd.transform(W)

    draw_interactive(W[:,0], W[:,1], words)

