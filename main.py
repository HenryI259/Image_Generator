from gather_data import getData
from network import network
import pickle as pkl
import gzip
import numpy as np

def main():
    network1 = network([1, 7500], 'heaven4')
    arrays = getData('heaven', 10000, record=True)
    image = network1.generateImage()
    image.save('HeavenI.png')
    network1.train([arrays[0] for i in range(50000)], 5000, 1, cycles=1, record=True, saveData=True)
    image = network1.generateImage()
    image.save('HeavenF.png')

if __name__ == '__main__':
    main()