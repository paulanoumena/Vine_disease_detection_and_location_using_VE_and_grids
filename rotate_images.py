import cv2

image = cv2.imread('data/230526_rbg.jpg')

(height, width) = image.shape[:2]
center = (width / 2, height / 2)
angle = -192
scale = 1.0
matrix = cv2.getRotationMatrix2D(center, angle, scale)
rotated_by_matrix = cv2.warpAffine(image, matrix, (width, height))

cv2.imwrite('data/230526_rgb_rotated.jpg', rotated_by_matrix)