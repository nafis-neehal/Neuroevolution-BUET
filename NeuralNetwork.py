#import libraries and stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#initialize the network

class Layer:
	def __init__(self, layer_parameter):
		if(layer_parameter[1]!='none'): 
			self.activations_for_this_layer = np.zeros(layer_parameter[0])
			self.number_of_nodes = layer_parameter[0]
			self.activation_function = layer_parameter[1]
			self.weights_from_previous_layer = layer_parameter[2]
			self.biases_for_this_layer = layer_parameter[3]
	
		
class NeuralNetwork:
	#constructor
	def __init__(self, network_parameter):
		self.total_weights_and_biases = 0
		self.number_of_hidden_layers = network_parameter[0]
		self.input_units = network_parameter[1]
		self.hidden_units = network_parameter[2]
		self.output_units = network_parameter[3]
		self.number_of_samples = network_parameter[4]
		self.hidden_layer_activation = network_parameter[5]
		self.layers = []
		self.final_output = 0
		
	#activation functions
	def ReLU(self, x):
		return np.maximum(x, 0)

	def DReLU(self, x):
		x[x<=0] = 0
		x[x>0] = 1
		return x
	
	def sigmoid(self, x):
		#return np.divide(1, np.add(1, np.exp(np.negative(x))))
		result = 1.0 / (1.0 + np.exp(-1.0 * x))
		return result
	
	def Dsigmoid(self, x):
		return sigmoid(x) * (1-sigmoid(x))
	
	def softmax(self, x):
		exp = np.exp(x)
		if isinstance(x[0], np.ndarray):
			return exp/np.sum(exp, axis=1, keepdims=True)
		else:
			return exp/np.sum(exp, keepdims=True)
	
	def tanh(self, x):
		return np.divide(np.exp(x)-np.exp(np.negative(x)) , np.exp(x)+np.exp(np.negative(x)))
	
	def Dtanh(self, x):
		return (1-self.tanh(x)**2)
	
	#network structure define
	def structure_define(self):
		
		#create input layer
		input_layer = Layer([self.input_units, 'none', [], []])
		self.layers.append(input_layer)

		#hidden layer activation function mapping
		if self.hidden_layer_activation == [1, 0, 0]:
			self.hidden_layer_activation = 'ReLU'
		elif self.hidden_layer_activation == [0, 1, 0]:
			self.hidden_layer_activation = 'tanh'
		elif self.hidden_layer_activation == [0, 0, 1]:
			self.hidden_layer_activation = 'sigmoid'

		#create hidden layer 1
		np.random.seed(1)
		hidden_layer_1_weights = np.random.randn(self.hidden_units, self.input_units) * 0.01 
		hidden_layer_1_biases = np.random.randn(self.hidden_units, 1)
		hidden_layer_1 = Layer([self.hidden_units, self.hidden_layer_activation, hidden_layer_1_weights, hidden_layer_1_biases])
		self.layers.append(hidden_layer_1)

		#create hidden layer 2
		np.random.seed(1)
		hidden_layer_2_weights = np.random.randn(self.hidden_units, self.hidden_units) * 0.01
		hidden_layer_2_biases = np.random.randn(self.hidden_units, 1)
		hidden_layer_2 = Layer([self.hidden_units, self.hidden_layer_activation, hidden_layer_2_weights, hidden_layer_2_biases])
		self.layers.append(hidden_layer_2)

		#create output layer
		np.random.seed(1)
		output_layer_weights = np.random.randn(self.output_units, self.hidden_units) * 0.01
		output_layer_biases = np.random.randn(self.output_units, 1)
		output_layer = Layer([self.output_units, 'Sigmoid', output_layer_weights, output_layer_biases])
		self.layers.append(output_layer)

		#total weights and biases
		self.total_weights_and_biases = (self.hidden_units * self.input_units) + (self.hidden_units * self.hidden_units) + \
			(self.output_units * self.hidden_units) + (2 * self.hidden_units) + self.output_units
			
	def forward_propagation(self, X): 
		
		# Hidden Layer 1
		self.W_1 = self.layers[1].weights_from_previous_layer
		self.B_1 = self.layers[1].biases_for_this_layer
		self.Z_1 = np.dot(self.W_1, X) + self.B_1
		if self.layers[1].activation_function=='ReLU': 
			self.A_1 = self.ReLU(self.Z_1)
		elif self.layers[1].activation_function=='tanh': 
			self.A_1 = self.tanh(self.Z_1)
		elif self.layers[1].activation_function=='sigmoid': 
			self.A_1 = self.sigmoid(self.Z_1)
		
		#Hidden Layer 2
		self.W_2 = self.layers[2].weights_from_previous_layer
		self.B_2 = self.layers[2].biases_for_this_layer
		self.Z_2 = np.dot(self.W_2, self.A_1) + self.B_2
		self.A_2 = self.ReLU(self.Z_2)
		if self.layers[2].activation_function=='ReLU': 
			self.A_2 = self.ReLU(self.Z_2)
		elif self.layers[2].activation_function=='tanh': 
			self.A_2 = self.tanh(self.Z_2)
		elif self.layers[2].activation_function=='sigmoid': 
			self.A_2 = self.sigmoid(self.Z_2)
		
		#Output Layer
		self.W_3 = self.layers[3].weights_from_previous_layer
		self.B_3 = self.layers[3].biases_for_this_layer
		self.Z_3 = np.dot(self.W_3, self.A_2) + self.B_3
		self.A_3 = self.sigmoid(self.Z_3)

		#Return Result
		return self.A_3
	
	def calculate_network_loss(self, Y):
		# loss = np.sum((-1) * (np.dot(Y, np.log(self.A_3.T))) + np.dot((1-Y),(np.log(1-self.A_3.T))))
		# loss = abs(loss / self.number_of_samples)
		loss = 0
		for i in range(self.output_units):
			arr1 = self.A_3[i].T
			arr1[arr1 == 0] = 0.000000000001 #remove zeros to avoid log 0 which will produce nan
			loss_1 = np.dot(Y[i], np.log(arr1))
			arr2 = 1-self.A_3[i].T
			arr2[arr2 == 0] = 0.000000000001 #remove zeros to avoid log 0 which will produce nan
			loss_2 = np.dot((1-Y[i]),np.log(arr2))
			loss_3 = -loss_1 - loss_2
			loss_3 = abs(loss_3 / Y.shape[1])
			loss = loss + loss_3 
		return loss
		
	def backward_propagation(self, X, Y):
		
		#Output Layer
		self.dZ_3 = self.A_3 - Y
		self.dW_3 = (1/self.number_of_samples) * np.dot(self.dZ_3, self.A_2.T)
		self.dB_3 = (1/self.number_of_samples) * np.sum(self.dZ_3, axis=1, keepdims=True)
		
		#Hidden Layer 2
		self.dZ_2 = np.multiply(np.dot(self.W_3.T, self.dZ_3), self.DReLU(self.Z_2))
		self.dW_2 = (1/self.number_of_samples) * np.dot(self.dZ_2, self.A_1.T)
		self.dB_2 = (1/self.number_of_samples) * np.sum(self.dZ_2, axis=1, keepdims=True)
		
		#Hidden Layer 1
		self.dZ_1 = np.multiply(np.dot(self.W_2.T, self.dZ_2), self.DReLU(self.Z_1))
		self.dW_1 = (1/self.number_of_samples) * np.dot(self.dZ_1, X.T)
		self.dB_1 = (1/self.number_of_samples) * np.sum(self.dZ_1, axis=1, keepdims=True)
		
		#Weight Update
		self.layers[1].weights_from_previous_layer = self.layers[1].weights_from_previous_layer - 0.01 * self.dW_1
		self.layers[1].biases_for_this_layer = self.layers[1].biases_for_this_layer - 0.01 * self.dB_1
		
		self.layers[2].weights_from_previous_layer = self.layers[2].weights_from_previous_layer - 0.01 * self.dW_2
		self.layers[2].biases_for_this_layer = self.layers[2].biases_for_this_layer - 0.01 * self.dB_2
		
		self.layers[3].weights_from_previous_layer = self.layers[3].weights_from_previous_layer - 0.01 * self.dW_3
		self.layers[3].biases_for_this_layer = self.layers[3].biases_for_this_layer - 0.01 * self.dB_3