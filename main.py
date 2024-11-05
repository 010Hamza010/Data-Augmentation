from os import path
from glob import glob

import numpy as np

import aug

LOCAL_DIR = path.dirname(path.realpath(__file__))

DATASET = path.join(LOCAL_DIR, "v1")

folders = ["train", "valid", "test"]

augmentation_number = 20

def chunks(xs, n):
	n = max(1, n)
	return [xs[i:i+n] for i in range(0, len(xs), n)]


for folder in folders:
	labels_dir = path.join(LOCAL_DIR, f"v1/{folder}/labels")
	images_dir = path.join(LOCAL_DIR, f"v1/{folder}/images")

	labels = list(glob(path.join(labels_dir, "*.txt")))

	print(labels)

	for label in labels:
		print(label)
		image = glob(path.join(images_dir, path.basename(label)[:-4]) + "*")[0]

		with open(label) as f:
			c = f.read()

		points = {}

		for i in c.split("\n"):
			points[i.split(" ")[0]] = chunks(i.split(" ")[1:], 2)


		points_all = []
		points_lens = {}

		for k,v in points.items():
			points_all += v
			points_lens[k]=len(v)

		imgs, pts = aug.Generate(image, np.array(points_all, dtype=np.float64), augmentation_number)

		for index in range(augmentation_number):

			augmented_image = path.join(images_dir, f"{index}-{path.basename(image)}")
			augmented_label = path.join(labels_dir, f"{index}-{path.basename(label)}")

			pt = list(pts[index])

			augmented_label_text = []

			for k,v in points_lens.items():

				text = []

				for i in pt[:v]:
					text.append(" ".join([str(x) for x in i]))

				augmented_label_text.append(f"{int(k)} {' '.join(text)}")
				#print(augmented_label_text)
				del pt[:v]

			#write to label & img

			with open(augmented_label, "w") as f:
				f.write("\n".join(augmented_label_text))

			imgs[index].save(augmented_image)

#print(imgs, pts)