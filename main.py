from gather_data import getData
from network import network
import pickle as pkl
import gzip
import numpy as np

def main():
    imageSize = (4000, 4000)
    
    imageNetwork = network([1, imageSize[0]*imageSize[1]*3])
    arrays = getData('space', 10, imageSize, record=True)
    image = imageNetwork.generateImage(imageSize)
    imageNetwork.train(arrays[0:7], 2500, 7, cycles=1, record=True)
    image = imageNetwork.generateImage(imageSize)
    image.save('image.png')

if __name__ == '__main__':
    main()