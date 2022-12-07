import csv
import cv2
import numpy as np
import random
import math

shapes = ["circle", "square", "triangle", "star"]
colors = ["red", "blue", "green", "yellow"]
image_info = {}


def get_bgr(color):
    if color == "red":
        bgr = (0, 0, 255)
    elif color == "green":
        bgr = (0, 255, 0)
    elif color == "blue":
        bgr = (255, 0, 0)
    elif color == "yellow":
        bgr = (0, 255, 255)

    return bgr


counter = 0

# SQUARE
for i in range(250):
    for color in colors:
        counter += 1
        img = np.ones((224, 224, 3), np.uint8)
        center_x = random.randint(0, 223)
        center_y = random.randint(0, 223)
        center = (center_x, center_y)
        size_val = random.randint(0, 223)
        size = (size_val, size_val)
        angle = random.randint(0, 89)

        rot_rectangle = (center, size, angle)
        box = cv2.boxPoints(rot_rectangle)
        box = np.int0(box)  # Convert into integer values
        bgr = get_bgr(color)
        rectangle = cv2.drawContours(img, [box], 0, bgr, -1)

        filename = 'images/' + str(counter) + '.jpg'
        image_info[filename] = [color, "square"]
        cv2.imwrite(filename, img)

# CIRCLE
for i in range(250):
    for color in colors:
        counter += 1
        img = np.ones((224, 224, 3), np.uint8)
        center_x = random.randint(0, 223)
        center_y = random.randint(0, 223)
        center = (center_x, center_y)
        radius = random.randint(0, 111)
        thickness = -1
        bgr = get_bgr(color)
        circle = cv2.circle(img, center, radius, bgr, thickness)

        filename = 'images/' + str(counter) + '.jpg'
        image_info[filename] = [color, "circle"]
        cv2.imwrite(filename, img)


# TRIANGLE
def get_third_point(x1, y1, x2, y2):
    #express coordinates of the point (x2, y2) with respect to point (x1, y1)
    dx = x2 - x1
    dy = y2 - y1

    alpha = 60./180 * math.pi
    #rotate the displacement vector and add the result back to the original point
    xp = x1 + math.cos( alpha)*dx + math.sin(alpha)*dy
    yp = y1 + math.sin(-alpha)*dx + math.cos(alpha)*dy

    return int(xp), int(yp)


for i in range(250):
    for color in colors:
        counter += 1
        img = np.ones((224, 224, 3), np.uint8)
        x1 = random.randint(0, 223)
        y1 = random.randint(0, 223)
        x2 = random.randint(0, 223)
        y2 = random.randint(0, 223)
        p1 = (x1, y1)
        p2 = (x2, y2)
        p3 = get_third_point(x1, y1, x2, y2)
        triangle_cnt = np.array([p1, p2, p3])
        bgr = get_bgr(color)
        triangle = cv2.drawContours(img, [triangle_cnt], 0, bgr, -1)

        filename = 'images/' + str(counter) + '.jpg'
        image_info[filename] = [color, "triangle"]
        cv2.imwrite(filename, img)


# STARS

for i in range(250):
    for color in colors:
        counter += 1
        pentagon = []
        img = np.ones((224, 224, 3), np.uint8)
        R = random.randint(0, 100) # radius
        shift_x = random.randint(50, 150)
        shift_y = random.randint(50, 150)
        rotation = random.randint(90, 126)
        for n in range(0,5):
          x = int(R*math.cos(math.radians(rotation+n*72))) + shift_x
          y = int(R*math.sin(math.radians(rotation+n*72))) + shift_y
          pentagon.append((x,y))
        penta = np.array(pentagon)
        bgr = get_bgr(color)
        pentag = cv2.drawContours(img, [penta], 0, bgr, -1)
        pentagon.append(pentagon[0])
        star = np.array(pentagon)
        for i in range(len(star) - 1):
            x1, y1 = star[i][0], star[i][1]
            x2, y2 = star[i + 1][0], star[i + 1][1]
            p1 = (x1, y1)
            p2 = (x2, y2)
            p3 = get_third_point(x1, y1, x2, y2)
            triangle_cnt = np.array([p1, p2, p3])
            bgr = get_bgr(color)
            triangle = cv2.drawContours(img, [triangle_cnt], 0, bgr, -1)

        filename = 'images/' + str(counter) + '.jpg'
        image_info[filename] = [color, "star"]
        cv2.imwrite(filename, img)


with open('labels.csv', 'w') as f:
    w = csv.DictWriter(f, image_info.keys())
    w.writeheader()
    w.writerow(image_info)


