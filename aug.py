import numpy as np
from PIL import Image, ImageDraw
from random import randint, choices

import cv2

from os import path, mkdir

from functions import *


LOCAL_DIR = path.dirname(path.realpath(__file__))

OUTPUT_FOLDER = path.join(LOCAL_DIR, "augmented")


def apply_fill_mode(image: np.ndarray, fill_mode='nearest', constant_value=0) -> np.ndarray:

	return image
	"""Applies fill mode to an image.

	Args:
		image (np.ndarray): The input image.
		fill_mode (str): The fill mode to use. Options: 'constant', 'nearest', 'reflect', 'wrap'.
		constant_value (int): The constant value to fill if 'constant' is chosen.

	Returns:
		np.ndarray: The image after applying the fill mode.
	"""
	if fill_mode == 'constant':
		return np.full_like(image, constant_value)  # Fill with constant value

	elif fill_mode == 'nearest':
		return cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

	elif fill_mode == 'reflect':
		return cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

	elif fill_mode == 'wrap':
		return cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_WRAP)

	else:
		raise ValueError(f"Unknown fill_mode: {fill_mode}")


def filter_points_within_image(points: np.ndarray, width: int, height: int) -> np.ndarray:
	for i in points:
		if (i[0] < 0):
			i[0] = 0
		elif (i[0] > width):
			i[0] = width

		if (i[1] < 1):
			i[1] = 0
		elif (i[1] > height):
			i[1] = height

	return points

# Function to add dots at points
def add_dots(img: np.ndarray, points: np.ndarray, dot_color=(0, 255, 0), dot_radius=10) -> np.ndarray:
	image_pil = Image.fromarray(img)
	draw = ImageDraw.Draw(image_pil)

	fac=0

	colors = [
		(255,0,0),
		(0,255,0),
		(0,0,255),
		(255,255,255),
		(0,0,0),
		(255,0,255),
		(255,255,0)
	]
	
	for point in points:
		x, y = point
		# Draw a small circle (dot) at the (x, y) location
		dot_color = colors[fac%len(colors)]
		fac += 1

		draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=dot_color)
	
	return np.array(image_pil)


def Generate(src, points, n_output=20):
	image = Image.open(src)
	image_np = np.array(image)

	augmented_images = []
	augmented_points = []

	for i in range(n_output):

		#pts = np.array([[0.21574074074074073, 0.4233937397034596], [0.725925925925926, 0.4200988467874794], [0.7342592592592593, 0.4843492586490939], [0.18888888888888888, 0.4958813838550247], [0.6638888888888889, 0.7808896210873146], [0.27685185185185185, 0.7973640856672158]])*(image.width,image.height)

		#pts = np.array([[0.5081521739130435, 0.19806763285024154], [0.701766304347826, 0.213768115942029], [0.7038043478260869, 0.5458937198067633], [0.5217391304347826, 0.5060386473429952]])*(image.width,image.height)

		pts = points * (image.width,image.height)

		func_n = randint(5,10)
		#func_n = 2

		funcs = choices(FUNCTIONS, k=func_n)

		#funcs = []

		augmented_image = image_np

			
		for func in funcs:
			#print(func.__name__)
			augmented_image, pts = func(augmented_image, pts)

		#augmented_image, pts = random_shear(image_np, pts)

		augmented_image = fill_nearest(augmented_image)

		#augmented_image = add_dots(augmented_image, filter_points_within_image(pts, image.width, image.height))

		augmented_image_pil = Image.fromarray(augmented_image)

		augmented_images.append(augmented_image_pil)
		augmented_points.append(pts / (image.width,image.height))

		#augmented_image_pil.save(path.join(OUTPUT_FOLDER, f"{i}.jpg"))

	return augmented_images, augmented_points

def main():
	if not path.exists(OUTPUT_FOLDER):
		mkdir(OUTPUT_FOLDER)

	pts = np.array([[0.21574074074074073, 0.4233937397034596], [0.725925925925926, 0.4200988467874794], [0.7342592592592593, 0.4843492586490939], [0.18888888888888888, 0.4958813838550247], [0.6638888888888889, 0.7808896210873146], [0.27685185185185185, 0.7973640856672158]])

	Generate(path.join(LOCAL_DIR, "kart.png"), pts, 5)

if __name__ == '__main__':
	main()


#augmented_image_pil.show()  # To display the image
