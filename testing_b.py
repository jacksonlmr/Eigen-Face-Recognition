import numpy as np
import cv2
import argparse
from helpers import load_images, mahalanobis_dist, calc_error_b, get_top_r_labels
from plot import plot_roc_b
import os

# parse cmd line args 
parser = argparse.ArgumentParser()

# parser.add_argument('filename', help='Path to image file to identify')
parser.add_argument('test_data_path', help='Path to directory containing test data')
parser.add_argument('training_results_path', help='Path to folder in which training results were stored')
parser.add_argument('-t', '--infothreshold', type=float, default=.95, help='Proportion of information to preserve')
parser.add_argument('-r', '--rank', type=int, default=1, help='Threshold within which an identification will be considered correct')

args = parser.parse_args()

training_data_dir = "Faces_FA_FB/fa2_H"
test_data_dir = args.test_data_path
training_results_path = args.training_results_path
info_threshold = args.infothreshold
rank = args.rank

results_dir = os.path.join(training_results_path, f"results_{int(info_threshold*100)}")

# calculate num eigen faces to keep based on information threshold
eigen_values_all = np.load(os.path.join(training_results_path, "eigen_values.npy"))

total_variance = np.sum(eigen_values_all)
cumulative_variance = np.cumsum(eigen_values_all)
explained_variance_ratio = cumulative_variance / total_variance

num_eigenfaces = np.argmax(explained_variance_ratio > info_threshold) + 1
print(f"Num eigen faces kept: {num_eigenfaces}")

# load eigenfaces, coefs, and avg face with num requested in args
eigen_values = eigen_values_all[:num_eigenfaces]
eigen_vectors = np.load(os.path.join(training_results_path, "eigen_vectors.npy"))[:, :num_eigenfaces]
eigen_coef = np.load(os.path.join(training_results_path, "eigen_coef.npy"))[:, :num_eigenfaces]
avg_face = np.load(os.path.join(training_results_path, "avg_face.npy"))
training_labels = np.load(os.path.join(training_results_path, "training_labels.npy"))

testing_imgs_labels, testing_imgs, img_shape = load_images(test_data_dir)

# center testing imgs
testing_imgs_centered = testing_imgs - avg_face

# compute coef
test_coef = testing_imgs_centered @ eigen_vectors

# compute mahalanobis distance
distances = mahalanobis_dist(test_coef, eigen_coef, eigen_values)

# calculate error
# rejected if distance is greater than the threshold
tprs= []
fprs = []
for threshold in np.arange(0, 1.25, 0.01):
    print(threshold)
    tpr, fpr = calc_error_b(training_labels, distances, testing_imgs_labels, threshold)
    tprs.append(tpr)
    fprs.append(fpr)

plot_roc_b(tprs, fprs, "ROC Curve")