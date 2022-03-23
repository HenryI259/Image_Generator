from gather_data import getData
from network import network
import pickle as pkl
import gzip
import numpy as np

def main():
    imageSize = (750, 750)
    
    imageNetwork = network([1, imageSize[0]*imageSize[1]*3], 'natureNetwork')
    arrays = getData('nature', 10, imageSize, record=True)
    imageNetwork.train(arrays, 1000, len(arrays), cycles=1, record=True, saveData=True)
    image = imageNetwork.generateImage(imageSize)
    image.save('image.png')

if __name__ == '__main__':
    main()