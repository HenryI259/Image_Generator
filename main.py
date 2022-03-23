from gather_data import getData
from network import network
import pickle as pkl
import gzip
import numpy as np
import os.path

def main():
    imageSize = (1000, 1000)
    
    imageNetwork = network([1, imageSize[0]*imageSize[1]*3])
    arrays = getData('space', 5, imageSize, record=True)
    imageNetwork.train(arrays, 1000, len(arrays), cycles=1, record=True)
    image = imageNetwork.generateImage(imageSize)
    image.save('image.png')

def removeData(network, arrays, name):
    if network:
        os.remove(f'networks\{name}.pkl.gz')
    if arrays:
        os.remove(f'image_data\{name}.pkl.gz')

if __name__ == '__main__':
    main()