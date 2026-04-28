import numpy as np
import cv2
from pathlib import Path

test_data_path = Path("Faces_FA_FB/fa_H")

# get first file
test_data_iter = test_data_path.iterdir()
first_file = next(test_data_iter, None)
first_file_array = cv2.imread(first_file, 0)
img_shape = first_file_array.shape

# set first file flattened as the first row of training imgs
training_imgs_names = [first_file.name]
training_imgs = first_file_array.flatten()

# training_images will be an array where each row is the flattened pixels of an image
for file in test_data_iter:
    if file.is_file():
        training_imgs_names.append(file.name)
        img_array = cv2.imread(file, 0)
        img_array = img_array.flatten()

        training_imgs = np.vstack((training_imgs, img_array))

# get average face
training_avg = np.mean(training_imgs, axis=0)
np.save("avg_face.npy", training_avg)

# visualize average face
avg_vis = np.reshape(training_avg, img_shape).astype(np.uint8)
cv2.imwrite("average_face.jpg", avg_vis)

# center data
training_imgs_centered = training_imgs - training_avg

# compute eigenvectors (using trick A_T*A instead of A*A_T)
a_t_a = training_imgs_centered @ training_imgs_centered.T

eigenvalues, eigenvectors = np.linalg.eigh(a_t_a)

# sort by largest
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# project to original cov matrix space
u_i = training_imgs_centered.T @ eigenvectors

# normalize vectors 
norms = np.linalg.norm(u_i, axis=0)
u_i = u_i / norms

np.save("eigen_vectors.npy", u_i)
# visualize eigenfaces
u_i_images = (255 * (u_i - u_i.min(axis=0)) / (u_i.max(axis=0) - u_i.min(axis=0))).astype(np.uint8)

for i in range(u_i_images.shape[1]):
    img_flat = u_i_images[:, i]
    img = np.reshape(img_flat, img_shape)
    cv2.imwrite(f"eigen_faces_imgs/ef_{i}.jpg", img)

# compute eigen coefficients
all_coef = training_imgs_centered @ u_i
np.save("eigen_coef.npy", all_coef)
