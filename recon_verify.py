import numpy as np
import cv2

input_face = cv2.imread("Faces_FA_FB/fa_H/00001_930831_fa_a.pgm", 0)
cv2.imwrite("recon_check.jpg", input_face)
avg_face = np.load("avg_face.npy")
eigen_vectors = np.load("eigen_vectors.npy")
eigen_coef = np.load("eigen_coef.npy")

eigen_coef_1 = eigen_coef[0, :]
recon_img =  (eigen_vectors @ eigen_coef_1) + avg_face

error = np.linalg.norm(input_face.flatten() - recon_img) / input_face.size
print(f"Reconstruction error: {error}")
recon_img = np.reshape(recon_img, input_face.shape)

recon_img_normalized = 255 * (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())

cv2.imwrite("recon_img.jpg", recon_img_normalized.astype(np.uint8))

print("\nVerifying eigen calculations for matrix: ")
test_matrix = np.array([[1, 2, 0], [2, 6, 1], [0, 1, 1]])
print(test_matrix)
values, vectors = np.linalg.eig(test_matrix)
print(f"Values: {values}")
print(f"Vectors: {vectors}")