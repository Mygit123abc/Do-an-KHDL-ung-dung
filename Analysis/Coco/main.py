from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import statistics
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from parralel import Multitask
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm
import pickle
import os

glove = api.load('glove-twitter-25')

def top_3_largest(data):
    top_3_values = [-float('inf'), -float('inf'), -float('inf')]
    top_3_indexes = [-1, -1, -1]

    for index, p in enumerate(data):
        if p > top_3_values[0]:
            top_3_values[2] = top_3_values[1]
            top_3_values[1] = top_3_values[0]
            top_3_values[0] = p

            top_3_indexes[2] = top_3_indexes[1]
            top_3_indexes[1] = top_3_indexes[0]
            top_3_indexes[0] = index
        elif p > top_3_values[1]:
            top_3_values[2] = top_3_values[1]
            top_3_values[1] = p

            top_3_indexes[2] = top_3_indexes[1]
            top_3_indexes[1] = index
        elif p > top_3_values[2]:
            top_3_values[2] = p
            top_3_indexes[2] = index

    return top_3_indexes

def plot_curve(ax, x, y, x_label, y_label, title, rotate=False):
    ax.plot(x, y)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.tick_params(axis='x', rotation=90)
    
def flatten(xss):
    return [x for xs in xss for x in xs]

def word_vec(word):
    try:
        return glove[word]
    except:
        return None

def short_term(term):
    avg = None
    
    words = term.split(' ')
    for word in words:
        if avg is None:
            avg = np.copy(word_vec(word))
        else:
            avg += word_vec(word)

    return avg / len(words)

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def calculate_metrics(tup):
    img_ids, worker_num = tup
    bar = tqdm(img_ids)
    bar.set_description_str(f'Worker {worker_num}')
    
    undefined_captions = []
    h_means_over_image_space = []
    h_max_sims_over_image_space = []

    for img_id in bar:
        ann_ids = instances_reader.getAnnIds(imgIds=img_id)
        anns = instances_reader.loadAnns(ids=ann_ids)
        captions = images_with_captions[img_id]

        means = []
        max_sims = []

        for ann in anns:
            cat_id = ann['category_id']
            cat_name = instances_reader.loadCats(ids=cat_id)[0]['name']
            cat_vec = short_term(cat_name)

            model = cat_models[cat_id]
            scores = model.transform(captions)
            scores = scores.toarray()

            fets = np.array(model.get_feature_names_out())
            tfidf_sorting = [top_3_largest(score) for score in scores]

            for index, sorting in enumerate(tfidf_sorting):
                if abs(sorting[0]) < 1e-5 or abs(sorting[1]) < 1e-5 or abs(sorting[2]) < 1e-5:
                    undefined_captions.append(captions[index])
                    continue # not enough 3 key words

                top_3_keys = fets[sorting]
                w1 = word_vec(top_3_keys[0])
                w2 = word_vec(top_3_keys[1])
                w3 = word_vec(top_3_keys[2])

                if w1 is None or w2 is None or w3 is None:
                    undefined_captions.append(captions[index])
                    continue # not enough 3 key words
                
                # print('passed!')
                s1 = max(cosine_similarity(w1, cat_vec), 0)
                s2 = max(cosine_similarity(w2, cat_vec), 0)
                s3 = max(cosine_similarity(w3, cat_vec), 0)

                mean = (s1 + s2 + s3) / 3
                max_s = max(s1, max(s2, s3))
                means.append(mean)
                max_sims.append(max_s)

        if len(means) != 0:
            h_means_over_image_space.append(statistics.harmonic_mean(means))
            h_max_sims_over_image_space.append(statistics.harmonic_mean(max_sims))
        else:
            with open('error.dat', 'a+') as f:
                f.write(f'{img_id}\n')
        
    return h_means_over_image_space, h_max_sims_over_image_space, undefined_captions

if __name__ == "__main__":
    caps_per_each_category = {}
    captions_reader = COCO(os.path.join("coco2014", "annotations" , "captions_train2014.json"))
    instances_reader = COCO(os.path.join("coco2014", "annotations", "instances_train2014.json"))
    ann_ids = captions_reader.getAnnIds()
    anns = captions_reader.loadAnns(ids=ann_ids)
    print(f'Total captions: {len(anns)}')

    images_with_captions = {}
    bar = tqdm(anns)
    for ann in bar:
        img_id = ann['image_id']
        if images_with_captions.get(img_id, None) is None:
            images_with_captions[img_id] = []
        images_with_captions[img_id].append(ann['caption'])

    print('Building captions per category')

    bar = tqdm(images_with_captions)
    for img_id in bar:
        ann_ids = instances_reader.getAnnIds(imgIds=img_id)
        anns = instances_reader.loadAnns(ids=ann_ids)
        captions = images_with_captions[img_id]
        for ann in anns:
            cat_id = ann['category_id']
            if caps_per_each_category.get(cat_id) is None:
                caps_per_each_category[cat_id] = []
            caps_per_each_category[cat_id] += captions

    cat_models = {}

    print('Building tf-idf models...')
    bar = tqdm(caps_per_each_category)
    corpus_sizes = []
    names = []

    for cat_id in bar:
        captions = caps_per_each_category[cat_id]
        names.append(instances_reader.loadCats(ids=cat_id)[0]['name'])

        model = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stopwords.words('english'))
        model.fit(captions)
        cat_models[cat_id] = model

        corpus_sizes.append(len(captions))

    fig, axis = plt.subplots(1, 1, figsize=(10, 10))
    fig.tight_layout()
    plot_curve(axis, names, corpus_sizes, 'Category id', 'Corpus size', 'Corpus size/category')

    fig.savefig('corpuses.png', bbox_inches='tight')

    print('Check similarity...')
    indices = np.append(np.arange(0, len(images_with_captions), len(images_with_captions) // 12), 
              [len(images_with_captions)])
    
    tasks = []
    keys = list(images_with_captions.keys())

    for i in range(len(indices) - 1):
        tasks.append((keys[indices[i]:indices[i + 1]], i))

    output = Multitask.task(tasks, calculate_metrics)
    h_means_over_image_space, h_max_sims_over_image_space, ignored_captions = [list(i) for i in zip(*output)]

    h_means_over_image_space = flatten(h_means_over_image_space)
    h_max_sims_over_image_space = flatten(h_max_sims_over_image_space)
    ignored_captions = flatten(ignored_captions)

    with open('h_means_over_image_space.data', 'wb') as f:
        pickle.dump(h_means_over_image_space, f)

    with open('h_max_sims_over_image_space.data', 'wb') as f:
        pickle.dump(h_max_sims_over_image_space, f)

    with open('ignored.data', 'wb') as f:
        pickle.dump(ignored_captions, f)
    
    fig, axes = plt.subplots(1, 2)

    plot_curve(axes[0], np.arange(0, len(h_means_over_image_space)), h_means_over_image_space, 
               'Similarity mean', 'Image id', 'Harmonic mean of similarity means')
    plot_curve(axes[1], np.arange(0, len(h_max_sims_over_image_space)), h_max_sims_over_image_space,
               'Max', 'Image id', 'Harmonic mean of max similiarity')
    
    fig.tight_layout()
    
    print(f'Ignored {len(ignored_captions)} captions')

    plt.show()