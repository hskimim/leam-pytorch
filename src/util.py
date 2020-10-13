import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
from sklearn.manifold import TSNE



def similar_by_word(trained_embed, lemma_dict, word, topn=10):
    """
    search most similar word by using trained word embedding(V)
    return type is similar with gensim.models.Word2Vec.wv.similar_by_word
    """
    q_idx = lemma_dict[word]
    reversed_lemma_dict = {v: k for k, v in lemma_dict.iteritems()}

    normalized_embed = trained_embed / torch.norm(trained_embed, p=2, dim=1)\
        .unsqueeze(1).repeat(1, trained_embed.shape[1])
    normalized_embed[~torch.isfinite(normalized_embed)] = 0

    normalized_q_embed = normalized_embed[q_idx]

    cosine_sim = torch.matmul(normalized_embed, normalized_q_embed.unsqueeze(1)).reshape(-1)

    q_where = torch.argsort(cosine_sim)[-topn:].cpu().data.numpy()
    cosine_sim = cosine_sim.cpu().data.numpy()
    ls = []
    for q in q_where:
        ls.append((reversed_lemma_dict[q], cosine_sim[q]))
    return ls[:-1] # except query word

def visualize_tsne(model, X, y) :

    V = model.compat_model.word_embedding(X).cpu().data.numpy()
    V = V.mean(1)
    C = y.argmax(1).cpu().data.numpy()

    tsne = TSNE(random_state=0, perplexity=10)

    reduced_mat = tsne.fit_transform(V)
    label = np.unique(C)
    ys = [i + i ** 2 for i in range(y.shape[1])]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for y, c in zip(label, colors):
        print(y, c)
        x = reduced_mat[C == y]
        plt.scatter(x[:, 0], x[:, 1], color=c)
    plt.show()