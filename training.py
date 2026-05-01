import numpy as np
import cv2
from pathlib import Path
from helpers import load_images
import os

training_data_path = Path("Faces_FA_FB/fa_H")
training_results_path = "experiment_a_results"

training_imgs_labels, training_imgs, img_shape = load_images(training_data_path)
np.save(os.path.join(training_results_path, "training_labels.npy"), np.array(training_imgs_labels))

# get average face
training_avg = np.mean(training_imgs, axis=0)
np.save(os.path.join(training_results_path, "avg_face.npy"), training_avg)

# visualize average face
avg_vis = np.reshape(training_avg, img_shape).astype(np.uint8)
cv2.imwrite(os.path.join(training_results_path, "average_face.jpg"), avg_vis)

# center data
training_imgs_centered = training_imgs - training_avg

# compute eigenvectors (using trick A_T*A instead of A*A_T)
a_t_a = training_imgs_centered @ training_imgs_centered.T

eigenvalues, eigenvectors = np.linalg.eig(a_t_a)

# sort by largest
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# project to original cov matrix space
u_i = training_imgs_centered.T @ eigenvectors_sorted

# normalize vectors 
norms = np.linalg.norm(u_i, axis=0)
u_i = u_i / norms

np.save(os.path.join(training_results_path, "eigen_values.npy"), eigenvalues_sorted)
np.save(os.path.join(training_results_path, "eigen_vectors.npy"), u_i)
# visualize eigenfaces
u_i_images = (255 * (u_i - u_i.min(axis=0)) / (u_i.max(axis=0) - u_i.min(axis=0))).astype(np.uint8)

for i in range(u_i_images.shape[1]):
    img_flat = u_i_images[:, i]
    img = np.reshape(img_flat, img_shape)
    cv2.imwrite(os.path.join(training_results_path, f"eigen_faces_imgs/ef_{i}.jpg"), img)

# compute eigen coefficients, each row is the coef associated with that image
all_coef = training_imgs_centered @ u_i
np.save(os.path.join(training_results_path, "eigen_coef.npy"), all_coef)
