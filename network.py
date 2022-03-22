import random
import numpy as np
import pickle
import gzip
import os.path
from PIL import Image

class network():
  def __init__(self, layers, savedNetwork=None):
    self.savedNetwork = savedNetwork
    if savedNetwork and os.path.exists(f"Networks/{savedNetwork}.pkl.gz"):
      f = gzip.open(f"Networks/{savedNetwork}.pkl.gz", 'rb')
      self.layers, self.activators, self.weights, self.biases = pickle.load(f, encoding='latin1')
      f.close()
      self.length = len(self.layers)
    else:
      self.layers = layers
      self.length = len(layers)
      self.biases = [np.random.randn(y, 1) for y in layers[1:]]
      self.weights = [np.random.randn(y, x)
                      for x, y in zip(layers[:-1], layers[1:])]
    self.activators = sigmoid(np.random.rand(layers[0], 1))

    self.function = sigmoid
    self.functionDerivative = sigmoidDerivative

  def forward(self):
    activation = self.activators
    for i in range(self.length-1):
      activation = self.function(np.dot(self.weights[i], activation) + self.biases[i])
    return activation

  def generateImage(self, imageSize):
      array = np.array([255-int(x*255) for x in self.forward()]).reshape(imageSize[0], imageSize[1], 3)
      return Image.fromarray(array.astype(np.uint8))

  def cost(self, array):
      return np.sum((array-self.forward())**2)

  def generateMiniBatches(self, trainingData, size):
    return [trainingData[x*size:(x+1)*size] for x in range(len(trainingData)//size)]

  def train(self, trainingData, learningRate, miniBatchSize, cycles=1, record=False, saveData=False):
    cost = self.cost(trainingData[0])
    for cycle in range(cycles):
      for m, minibatch in enumerate(self.generateMiniBatches(trainingData, miniBatchSize)):
        deltaA = np.zeros(self.layers[0]).reshape(self.layers[0], 1)
        deltaW = [np.zeros(y*x).reshape(y, x)
                  for x, y in zip(self.layers[:-1], self.layers[1:])]
        deltaB = [np.zeros(y).reshape(y, 1) for y in self.layers[1:]]
        for i, x in enumerate(minibatch):
          if record:
            print(f'Dataset: {i + (m*miniBatchSize)}')
          da, dw, db = self.backprop(x)
          deltaA = np.add(deltaA, da)
          deltaW = np.add(deltaW, dw)
          deltaB = np.add(deltaB, db)
        self.activators = np.array([a-(na*learningRate/miniBatchSize) for a, na in zip(self.activators, deltaA)])
        self.weights = [w-(nw*learningRate/miniBatchSize) for w, nw in zip(self.weights, deltaW)]
        self.biases = [b-(nb*learningRate/miniBatchSize) for b, nb in zip(self.biases, deltaB)]

        if m%10 == 0 and record:
            print(self.cost(minibatch[0])-cost)
            cost = self.cost(minibatch[0])
        
        if m%500 == 0 and saveData and self.savedNetwork:
          self.saveNetwork(self.savedNetwork)
    if saveData and self.savedNetwork:
      self.saveNetwork(self.savedNetwork)

  def backprop(self, output):
    activations = [self.activators]
    zs = []
    deltaW = [np.zeros(y*x).reshape(y, x)
              for x, y in zip(self.layers[:-1], self.layers[1:])]
    deltaB = [np.zeros(y).reshape(y, 1) for y in self.layers[1:]]
    for layer in range(self.length-1):
      z = np.dot(self.weights[layer], activations[layer]) + self.biases[layer]
      zs.append(z)
      activations.append(self.function(z))
    for i in range(len(activations)):
      layer = len(activations)-i-1
      if i == 0:
        a = activations[-1]
        deltaA = [self.costDerivative(a[x], output[x]) for x in range(len(a))]
        deltaB[layer-1] = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[-1]))]).reshape(len(zs[-1]),1)
      else:
        dzda = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[layer]))]).reshape(len(zs[layer]),1)
        deltaW[layer] = np.dot(dzda, activations[layer].reshape(1,len(activations[layer])))
        deltaA = np.dot(self.weights[layer].transpose(), dzda)
        if layer != 0: deltaB[layer-1] = np.array([da * dz for da, dz in zip(deltaA, self.functionDerivative(zs[layer-1]))]).reshape(len(zs[layer-1]),1)
    return deltaA, deltaW, deltaB

  def costDerivative(self, x, y):
    return 2*(x-y)

  def saveNetwork(self, name):
    f = gzip.open(f'Networks/{name}.pkl.gz', 'w')
    pickle.dump((self.layers, self.activators, self.weights, self.biases), f)
    f.close()
    print('Network saved')
    

def sigmoid(x):
    # sigmoid function
    return 1.0/(1.0+np.exp(-x))

def sigmoidDerivative(x):
    # derivative of the sigmoid function
    return sigmoid(x)*(1-sigmoid(x))