from simple_image_download import simple_image_download as sid
from PIL import Image
import os.path
import shutil
import pickle as pkl
import gzip
import numpy as np

def getImages(image, amount):
	if not os.path.exists(f'simple_images\{image}'):
		response = sid.simple_image_download()
		response.download(image, amount+2)
		os.remove(f'simple_images\{image}\{image}_1.png')
		os.remove(f'simple_images\{image}\{image}_2.png')
		for i in range(amount):
			try:
				os.rename(f'simple_images\{image}\{image}_{i+3}.jpeg', f'simple_images\{image}\{i}.jpeg')
			except:
				os.rename(f'simple_images\{image}\{image}_{i+3}.png', f'simple_images\{image}\{i}.jpeg')

def imageToArray(fileName):
	image = Image.open(fileName)
	if image.size != (50, 50):
		image = image.resize((50, 50))
	array = np.asarray(image)
	try:
		return array.reshape(7500, 1) / 255
	except:
		pass

def getData(image, amount, saveData=True, record=False):
	if os.path.exists(f'image_data\{image}.pkl.gz'):
		f = gzip.open(f"image_data\{image}.pkl.gz", 'rb')
		arrays = pkl.load(f, encoding='latin1')
		f.close()
	else:
		getImages(image, amount)
		arrays = [imageToArray(f'simple_images\{image}\{i}.jpeg') for i in range(amount)]
		arrays = list(filter((None).__ne__, arrays))
		if saveData:
			f = gzip.open(f'image_data\{image}.pkl.gz', 'w')
			pkl.dump(arrays, f)
			f.close()
	return arrays
