import pickle
import cv2
from matplotlib import pyplot as plt
from extractor import extract_single_img
from rank import search_unseen_query
from dataset import Dataset
from PIL import Image
from cirtorch.datasets.datahelpers import default_loader

with open('data/gnd_oxford5k.pkl', 'rb') as f:
    data = pickle.load(f)

def demo(filename):
    # prepare some global variables 
    cache_dir = './tmp/oxford5k_resnet'
    dataset_name = 'oxford5k/'
    query_path = './data/query/oxford5k_resnet_glob.npy'
    gallery_path = './data/gallery/oxford5k_resnet_glob.npy'
    gnd_path = './data/gnd_oxford5k.pkl'
    truncation_size = 1000
    kq = 10
    kd = 50

    dataset = Dataset(query_path, gallery_path)
    queries, gallery = dataset.queries, dataset.gallery
    index_query_image = data['qimlist'].index(filename.replace('.jpg',''))
    query_input = queries[index_query_image].reshape(1, 2048)
    idx_result = search_unseen_query(query_input, gallery, truncation_size, kd, kq, cache_dir, gnd_path, gamma=3)
    result = ['oxford5k/'+data["imlist"][i]+'.jpg' for i in idx_result]
    return result

def demo_custom(filename):
    # prepare some global variables 
    cache_dir = './tmp/oxford5k_custom'
    dataset_name = 'oxford5k/'
    query_path = './data/query/oxford5k_custom_glob.npy'
    gallery_path = './data/gallery/gallery_oxford5k_custom_glob.npy'
    gnd_path = './data/gnd_oxford5k.pkl'
    truncation_size = 1000
    kq = 10
    kd = 50
    
    dataset = Dataset(query_path, gallery_path)
    queries, gallery = dataset.queries, dataset.gallery
    img_input = default_loader(dataset_name + filename)
    query_input = extract_single_img(img_input)
    idx_result = search_unseen_query(query_input, gallery, truncation_size, kd, kq, cache_dir, gnd_path, gamma=3)
    result = ['oxford5k/'+data["imlist"][i]+'.jpg' for i in idx_result]
    return result

# input_image = 'bodleian_000132.jpg'
# result = demo_custom(input_image)
# figure_size = 20
# index = 1
# plt.figure(figsize=(figure_size, figure_size))
# for i in result:
#     img = cv2.imread(i)
#     img = cv2.resize(img, (200, 200))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     plt.subplot(2, 5, index), plt.imshow(img)
#     plt.title(i), plt.xticks([]), plt.yticks([])

#     index += 1
# plt.subplots_adjust(hspace=-0.7, wspace=0.05)

# plt.savefig('result.jpg')

# query_img = cv2.imread('oxford5k/'+input_image)
# cv2.imwrite('query_image.jpg', query_img)