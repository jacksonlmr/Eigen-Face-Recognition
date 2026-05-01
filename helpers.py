import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def load_images(dir_path: str):
    # load test images
    test_data_path = Path(dir_path)

    # get first file
    test_data_iter = test_data_path.iterdir()
    first_file = next(test_data_iter, None)
    first_file_array = cv2.imread(first_file, 0)
    img_shape = first_file_array.shape

    # set first file flattened as the first row of training imgs
    testing_imgs_labels = [first_file.name]
    testing_imgs = first_file_array.flatten()

    # testing_images will be an array where each row is the flattened pixels of an image
    for file in test_data_iter:
        if file.is_file():
            testing_imgs_labels.append(file.name)
            img_array = cv2.imread(file, 0)
            img_array = img_array.flatten()

            testing_imgs = np.vstack((testing_imgs, img_array))

    return np.array(testing_imgs_labels), testing_imgs, img_shape

def mahalanobis_dist(test_coef, eigen_coef, eigen_values):
    diff_sqd = (test_coef[:, np.newaxis, :] - eigen_coef[np.newaxis, :, :]) ** 2
    distances = np.sum(diff_sqd / eigen_values, axis=2)

    return distances

def calc_error_a(testing_imgs_labels, top_r_labels):
    num_test_images = len(testing_imgs_labels)

    test_ids = np.array([l[:5] for l in testing_imgs_labels])
    train_ids = np.array([[l[:5] for l in row] for row in top_r_labels])

    matches = (train_ids == test_ids[:, np.newaxis])

    cumulative_matches = np.logical_or.accumulate(matches, axis=1)

    correct_counts_per_r = np.sum(cumulative_matches, axis=0)
    accuracies_array = (correct_counts_per_r / num_test_images) * 100

    return accuracies_array

def calc_error_b(training_labels, distances, testing_imgs_labels, threshold_tr):
    # get set of genuine people from the training set
    training_ids = set(label[:5] for label in training_labels)

    # Separate test images into genuines and intruders
    num_genuine = sum(1 for label in testing_imgs_labels if label[:5] in training_ids)
    num_intruder = len(testing_imgs_labels) - num_genuine

    # get min distances and their indices 
    min_distances = np.min(distances, axis=1)
    nearest_indices = np.argmin(distances, axis=1)

    # count true positives and false positives
    true_positives = 0
    false_positives = 0

    for i, test_label in enumerate(testing_imgs_labels):
        test_id = test_label[:5]
        accepted = min_distances[i] <= threshold_tr

        if test_id in training_ids:
            # TP if accepted and matched to the correct identity
            predicted_id = training_labels[nearest_indices[i]][:5]
            if accepted and predicted_id == test_id:
                true_positives += 1
        else:
            # FP if accepted 
            if accepted:
                false_positives += 1

    tpr = true_positives / num_genuine
    fpr = false_positives / num_intruder

    return tpr, fpr

def get_top_r_labels(training_labels, distances, rank):
    top_r_indices = np.argsort(distances, axis=1)[:, :rank]

    top_r_labels = training_labels[top_r_indices]

    return top_r_labels
