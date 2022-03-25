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

def imageToArray(fileName, imageSize):
	image = Image.open(fileName)
	if image.size != (imageSize[0], imageSize[1]):
		image = image.resize((imageSize[0], imageSize[1]))
	array = np.asarray(image)
	try:
		return array.reshape(imageSize[0] * imageSize[1] * 3, 1) / 255
	except:
		pass

def getData(image, amount, imageSize):
	iSize = ()
	a = -1
	if os.path.exists(f'image_data\{image}.pkl.gz'):
		f = gzip.open(f"image_data\{image}.pkl.gz", 'rb')
		arrays, iSize, a = pkl.load(f, encoding='latin1')
		f.close()

	if iSize != imageSize or a < amount or not os.path.exists(f'image_data\{image}.pkl.gz'):
		getImages(image, amount)
		arrays = [imageToArray(f'simple_images\{image}\{i}.jpeg', imageSize) for i in range(amount)]
		arrays = list(filter((None).__ne__, arrays))
		f = gzip.open(f'image_data\{image}.pkl.gz', 'w')
		pkl.dump((arrays, imageSize, amount), f)
		f.close()
	return arrays[:amount]
