import cv2
import numpy as np

map_image = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
road_edges = cv2.Canny(blurred_image, 50, 150)

road_contours, _ = cv2.findContours(road_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

outlined_image = map_image.copy()
cv2.drawContours(outlined_image, road_contours, -1, (0, 0, 255), 2)
_, tree_mask = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)
tree_contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in tree_contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(outlined_image, (cx, cy), 5, (0, 75, 0), -1)

cv2.imshow("Outlined Features", outlined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
