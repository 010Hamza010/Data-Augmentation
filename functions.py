from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, generic_filter


import numpy as np
import cv2

def fill_nearest(image_np: np.ndarray) -> np.ndarray:
    """
    Fills empty (zero) pixels in a NumPy image using the nearest non-zero pixel.
    
    Args:
    image_np (np.ndarray): Input image as a NumPy array (H, W, C) for RGB images.
    
    Returns:
    np.ndarray: The image with empty spaces filled by the nearest neighbor.
    """
    # Create a mask where all zero pixels (fully black) are identified as empty
    mask = np.all(image_np == 0, axis=-1)  # Shape: (H, W)

    # Use distance transform to find the nearest non-zero pixel for each empty pixel
    distance, nearest_indices = distance_transform_edt(mask, return_indices=True)

    # Extract row and column indices for the nearest non-empty pixels
    row_indices, col_indices = nearest_indices[0].astype(np.int32), nearest_indices[1].astype(np.int32)

    # Initialize the filled image
    filled_image = image_np.copy()

    filled_image[mask] = image_np[row_indices[mask], col_indices[mask]]

    return filled_image

def fill_image_with_reflection(image):
    """
    Fills NaN values in a rotated image using reflected parts of the image.
    """
    # Create a mask of NaN values
    mask = np.all(image == 0, axis=-1)  # Shape: (H, W)

    # Fill NaN values by reflecting from valid regions
    filled_image = image.copy()
    
    # Horizontal reflection: fill from the left side
    if np.any(mask):
        filled_image[mask] = np.fliplr(filled_image)[mask]

    # If any NaN values remain, try vertical reflection: fill from the top side
    mask = np.isnan(filled_image)
    if np.any(mask):
        filled_image[mask] = np.flipud(filled_image)[mask]

    return filled_image

def apply_fill_mode(image: np.ndarray, fill_mode='nearest', constant_value=0) -> np.ndarray:
	print(fill_mode)

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
		return cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REPLICATE)

	elif fill_mode == 'reflect':
		return cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)

	elif fill_mode == 'wrap':
		return cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_WRAP)

	else:
		raise ValueError(f"Unknown fill_mode: {fill_mode}")



# 1. Flip horizontally and transform points
def random_flip_horizontal(image: np.ndarray, points: np.ndarray, prob=1) -> tuple:
	if np.random.rand() < prob:
		image = np.fliplr(image)
		width = image.shape[1]
		points[:, 0] = width - points[:, 0]  # Adjust x-coordinates
	return image, points

# 2. Flip vertically and transform points
def random_flip_vertical(image: np.ndarray, points: np.ndarray, prob=1) -> tuple:
	if np.random.rand() < prob:
		image = np.flipud(image)
		height = image.shape[0]
		points[:, 1] = height - points[:, 1]  # Adjust y-coordinates
	return image, points

# 3. Rotate image and transform points
def random_rotation(image: np.ndarray, points: np.ndarray, max_angle=90) -> tuple:
	angle = np.random.uniform(-max_angle, max_angle)
	#angle = 30
	h, w = image.shape[:2]

	# Rotate image
	image_pil = Image.fromarray(image)
	image_rotated = np.array(image_pil.rotate(angle, resample=Image.BILINEAR))

	# Calculate the center of the image
	cx, cy = w / 2, h / 2
	angle_rad = -np.deg2rad(angle)

	# Rotate each point
	new_points = []
	for x, y in points:
		# Translate points to origin (center the points)
		x_shifted = x - cx
		y_shifted = y - cy

		# Apply rotation matrix
		new_x = x_shifted * np.cos(angle_rad) - y_shifted * np.sin(angle_rad)
		new_y = x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)

		# Translate back to original center
		new_x += cx
		new_y += cy

		new_points.append([new_x, new_y])

	new_points = np.array(new_points)

	# Return the rotated image and the new (potentially out-of-bounds) points
	return image_rotated, new_points

# 4. Add Gaussian noise (doesn't affect points)
def add_random_gaussian_noise(image: np.ndarray, points: np.ndarray, mean=0, std=25) -> tuple:
	noise = np.random.normal(mean, std, image.shape)
	noisy_image = image + noise
	return np.clip(noisy_image, 0, 255).astype(np.uint8), points

# 5. Adjust brightness (doesn't affect points)
def random_brightness(image: np.ndarray, points: np.ndarray, brightness_range=(0.8, 1.2)) -> tuple:
	factor = np.random.uniform(*brightness_range)
	return np.clip(image * factor, 0, 255).astype(np.uint8), points

# 6. Adjust contrast (doesn't affect points)
def random_contrast(image: np.ndarray, points: np.ndarray, contrast_range=(0.8, 1.2)) -> tuple:
	factor = np.random.uniform(*contrast_range)
	mean = np.mean(image)
	return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8), points

# 7. Random crop and transform points
def random_crop(image: np.ndarray, points: np.ndarray, crop_size=(500, 500)) -> tuple:
	h, w, _ = image.shape
	ch, cw = crop_size
	start_h = np.random.randint(0, h - ch)
	start_w = np.random.randint(0, w - cw)

	# Crop image
	cropped_image = image[start_h:start_h + ch, start_w:start_w + cw]
	cropped_image_resized = np.array(Image.fromarray(cropped_image).resize((w, h), resample=Image.BILINEAR))
	
	# Adjust points
	points[:, 0] = np.clip(points[:, 0] - start_w, 0, w)
	points[:, 1] = np.clip(points[:, 1] - start_h, 0, h)
	
	return cropped_image_resized, points

# 8. Zoom and transform points
def random_zoom(image: np.ndarray, points: np.ndarray, zoom_range=(0.8, 1.2)) -> tuple:
	h, w, _ = image.shape
	zoom_factor = np.random.uniform(*zoom_range)

	zoom_factor = 0.3
	
	# Calculate new dimensions
	new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
	
	if zoom_factor < 1:  # Zooming out (pad the image)
		padding_h = (h - new_h) // 2
		padding_w = (w - new_w) // 2
		
		# Create a new padded image
		zoomed_image = np.pad(image, ((padding_h, h - new_h - padding_h), 
									  (padding_w, w - new_w - padding_w), 
									  (0, 0)), mode='constant', constant_values=0)
		
		# Adjust the points based on the padding and zoom factor
		new_points = points * zoom_factor
		new_points[:, 0] += padding_w  # Shift x-coordinates
		new_points[:, 1] += padding_h  # Shift y-coordinates

	else:  # Zooming in (crop the image)
		start_h = (new_h - h) // 2
		start_w = (new_w - w) // 2
		
		# Crop and resize the zoomed-in image
		zoomed_image = image[start_h:start_h + h, start_w:start_w + w]
		
		# Adjust the points based on the cropping
		new_points = points * zoom_factor
		new_points[:, 0] -= start_w  # Shift x-coordinates back
		new_points[:, 1] -= start_h  # Shift y-coordinates back

	# Ensure points are within image boundaries
	#new_points[:, 0] = np.clip(new_points[:, 0], 0, w)
	#new_points[:, 1] = np.clip(new_points[:, 1], 0, h)

	return np.array(Image.fromarray(zoomed_image).resize((w, h), resample=Image.BILINEAR)), new_points

# 9. Translate and transform points
def random_translation(image: np.ndarray, points: np.ndarray, max_translation=(250, 250)) -> tuple:
	tx = np.random.randint(-max_translation[0], max_translation[0])
	ty = np.random.randint(-max_translation[1], max_translation[1])

	translated_image = np.roll(image, shift=(ty, tx), axis=(0, 1))
	
	# Handle padding
	if ty > 0:
		translated_image[:ty, :] = 0
	elif ty < 0:
		translated_image[ty:, :] = 0
	if tx > 0:
		translated_image[:, :tx] = 0
	elif tx < 0:
		translated_image[:, tx:] = 0

	# Translate points
	points[:, 0] = np.clip(points[:, 0] + tx, 0, image.shape[1])
	points[:, 1] = np.clip(points[:, 1] + ty, 0, image.shape[0])

	return translated_image, points

# 10. Invert colors (doesn't affect points)
def random_invert_colors(image: np.ndarray, points: np.ndarray, prob=1) -> tuple:
	if np.random.rand() < prob:
		return 255 - image, points
	return image, points

def random_shear(image: np.ndarray, points: np.ndarray, shear_range=(-0.2, 0.2)) -> tuple:
	shear_factor = np.random.uniform(*shear_range)

	h, w = image.shape[:2]

	# Create shear transformation matrix
	M = np.array([[1, shear_factor, 0], [0, 1, 0]])
	sheared_image = cv2.warpAffine(image, M, (w, h))

	# Update point positions
	new_points = np.array([
		[x + shear_factor * y, y] for x, y in points
	])
	
	return sheared_image, new_points

def random_perspective_transform(image: np.ndarray, points: np.ndarray, transform_range=(-250,250)) -> tuple:
	h, w = image.shape[:2]

	# Define source points (corners of the image)
	src_points = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
	
	# Define random destination points
	dst_points = src_points + np.random.uniform(*transform_range, src_points.shape).astype(np.float32)

	# Get perspective transformation matrix
	M = cv2.getPerspectiveTransform(src_points, dst_points)

	# Apply perspective transform
	warped_image = cv2.warpPerspective(image, M, (w, h))

	# Update point positions
	new_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), M)
	return warped_image, new_points.reshape(-1, 2)

def random_noise(image: np.ndarray, points: np.ndarray, noise_factor=0.1) -> tuple:
	noise = np.random.randn(*image.shape) * 255 * noise_factor
	noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
	return noisy_image, points  # Points remain unchanged

def random_color_jitter(image: np.ndarray, points: np.ndarray) -> tuple:
	# Randomly change brightness
	brightness_factor = np.random.uniform(0.5, 1.5)
	image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

	# Randomly change contrast
	contrast_factor = np.random.uniform(0.5, 1.5)
	mean = np.mean(image)
	image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)

	return image, points  # Points remain unchanged




FUNCTIONS = [
	random_flip_horizontal,
	random_flip_vertical,
	random_rotation,
	add_random_gaussian_noise,
	random_brightness,
	random_contrast,
	#random_zoom,
	random_translation,
	#random_invert_colors,
	random_shear,
	random_perspective_transform,
	random_noise,
	random_color_jitter
]