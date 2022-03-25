from gather_data import getData
from network import network
import pickle as pkl
import gzip
import numpy as np
import os.path

def main():
    generateImage(
        imageName='nature',
        imageSize=(3000, 3000),
        dataAmount=5,
        record=True
    )

def generateImage(imageName='', imageSize=(100, 100), dataAmount=3, record=False, saveNetwork=False, saveImage=True):
    imageNetwork = network([1, imageSize[0]*imageSize[1]*3])
    arrays = getData(imageName, dataAmount, imageSize)
    imageNetwork.train(arrays, 1000, len(arrays), record=record, saveData=saveNetwork)
    image = imageNetwork.generateImage(imageSize)
    if saveImage: image.save(f'{imageName}.png')

def removeData(network, arrays, name):
    if network:
        os.remove(f'networks\{name}.pkl.gz')
    if arrays:
        os.remove(f'image_data\{name}.pkl.gz')

if __name__ == '__main__':
    main()