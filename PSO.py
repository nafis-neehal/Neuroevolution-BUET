import numpy as np
from random import randint
from NeuralNetwork import Layer, NeuralNetwork

class ParticleStructure():
	def __init__(self, nVar, seed, nn):
		#np.random.seed(seed)
		#self.position = np.random.randn(nVar, 1) * 0.1
		self.position = np.append(nn.layers[1].weights_from_previous_layer, nn.layers[2].weights_from_previous_layer)
		self.position = np.append(self.position, nn.layers[3].weights_from_previous_layer) 
		self.position = np.append(self.position, nn.layers[1].biases_for_this_layer)
		self.position = np.append(self.position, nn.layers[2].biases_for_this_layer)
		self.position = np.append(self.position, nn.layers[3].biases_for_this_layer)
		self.position = np.reshape(self.position, (np.size(self.position), 1))
		random_pos = randint(0, self.position.size-1)
		self.position[random_pos] = self.position[random_pos] + 0.1
		self.velocity = np.zeros((nVar, 1))
		self.cost = None
		self.best_position = self.position
		self.best_cost = self.cost

class Swarm():
	global_best_position = None
	global_best_cost = np.Inf
	difference_best = None
	X = None
	Y = None
	Xt = None
	yt = None
	
	def __init__(self, nn, X, Y, maxIt, nPop, Xv, yv):
		
		#problem statement
		self.global_best_position = np.zeros((nn.total_weights_and_biases, 1))
		self.difference_best = np.zeros((nn.total_weights_and_biases, 1))
		# self.nVar = np.size(nn.layers[1].weights_from_previous_layer) + \
		#                          np.size(nn.layers[2].weights_from_previous_layer) + \
		#                          np.size(nn.layers[3].weights_from_previous_layer) + \
		#                          np.size(nn.layers[1].biases_for_this_layer) + \
		#                          np.size(nn.layers[2].biases_for_this_layer) + \
		#                          np.size(nn.layers[3].biases_for_this_layer)
		self.nVar = nn.total_weights_and_biases
		
		#constriction coefficients
		self.kappa = 1
		self.phi1 = 2.05
		self.phi2 = 2.05
		self.phi = self.phi1 + self.phi2
		self.chi = (2 * self.kappa) / abs(2-self.phi- np.sqrt((self.phi**2) - (4 * self.phi)))

		#parameters of PSO
		self.X = X
		self.Y = Y
		self.Xv = Xv
		self.yv = yv
		self.maxIt = maxIt                              #numer of iteration
		self.nPop = nPop                                #population size
		self.particle = []                              #all particles
		self.w = self.chi                               #intertia coefficient
		self.wdamp = 0.99                               #Damping Ratio of Inertia Coeffieicnt
		self.c1 = self.chi * self.phi1                  #personal acceleration coefficient
		self.c2 = self.chi * self.phi2                  #social acceleration coefficient
	
	def cost_function_pso(self, nn, particle):
		#run one forward prop of neural network
		lim = np.size(nn.layers[1].weights_from_previous_layer)
		nn.layers[1].weights_from_previous_layer = np.reshape(particle.position[0:lim],(nn.hidden_units, nn.input_units))
		lim2 = lim + np.size(nn.layers[2].weights_from_previous_layer)
		nn.layers[2].weights_from_previous_layer = np.reshape(particle.position[lim:lim2],(nn.hidden_units, nn.hidden_units))
		lim3 = lim2 + np.size(nn.layers[3].weights_from_previous_layer)
		nn.layers[3].weights_from_previous_layer = np.reshape(particle.position[lim2:lim3],(nn.output_units, nn.hidden_units))
		lim4 = lim3 + np.size(nn.layers[1].biases_for_this_layer)
		nn.layers[1].biases_for_this_layer = np.reshape(particle.position[lim3:lim4],(nn.hidden_units,1))
		lim5 = lim4 + np.size(nn.layers[2].biases_for_this_layer)
		nn.layers[2].biases_for_this_layer = np.reshape(particle.position[lim4:lim5],(nn.hidden_units,1))
		lim6 = lim5 + np.size(nn.layers[3].biases_for_this_layer)
		nn.layers[3].biases_for_this_layer = np.reshape(particle.position[lim5:lim6],(nn.output_units, 1))
		nn.forward_propagation(self.X)
		cost = nn.calculate_network_loss(self.Y)
		
		return cost
		
	def initialize_swarm(self, nn):
		for i in range (self.nPop):
			
			#initialize a Particle and add it to swarm list
			self.particle.append(ParticleStructure(self.nVar, i,nn))
			
			#calculate the cost of a particle in current position
			self.particle[i].cost = self.cost_function_pso(nn, self.particle[i])
			
			#update self best
			self.particle[i].best_position = self.particle[i].position
			self.particle[i].best_cost = self.particle[i].cost
			
			#update global best for whole swarm
			if self.particle[i].best_cost < self.global_best_cost:
				self.global_best_position = self.particle[i].best_position
				self.global_best_cost = self.particle[i].best_cost

	def pso_loop(self, nn, np_loss, validation_loss):
		for i in range(self.maxIt):
			for j in range(self.nPop):
				#update velocity
				self.particle[j].velocity = self.w * self.particle[j].velocity
				self.particle[j].velocity = self.particle[j].velocity\
											+ self.c1 * np.multiply(np.random.randn(self.nVar,1) * 0.1,\
											  (self.particle[j].best_position - self.particle[j].position))
				self.particle[j].velocity = self.particle[j].velocity\
											+ self.c2 * np.multiply(np.random.randn(self.nVar,1) * 0.1,\
											  (self.global_best_position - self.particle[j].position))
				#update particle position
				self.particle[j].position = self.particle[j].position + self.particle[j].velocity
				
				#calculate new particle cost
				self.particle[j].cost     = self.cost_function_pso(nn, self.particle[j])

				#update particle best 
				if self.particle[j].cost < self.particle[j].best_cost:
					self.particle[j].best_position = self.particle[j].position
					self.particle[j].best_cost = self.particle[j].cost
					#update global best
					if self.particle[j].best_cost < self.global_best_cost:
						self.global_best_position = self.particle[j].best_position
						self.global_best_cost = self.particle[j].best_cost
			
			np_loss[i] = self.global_best_cost

			#best network build for validation loss test
			lim = np.size(nn.layers[1].weights_from_previous_layer)
			nn.layers[1].weights_from_previous_layer = np.reshape(self.global_best_position[0:lim],(nn.hidden_units, nn.input_units))
			lim2 = lim + np.size(nn.layers[2].weights_from_previous_layer)
			nn.layers[2].weights_from_previous_layer = np.reshape(self.global_best_position[lim:lim2],(nn.hidden_units, nn.hidden_units))
			lim3 = lim2 + np.size(nn.layers[3].weights_from_previous_layer)
			nn.layers[3].weights_from_previous_layer = np.reshape(self.global_best_position[lim2:lim3],(nn.output_units, nn.hidden_units))
			lim4 = lim3 + np.size(nn.layers[1].biases_for_this_layer)
			nn.layers[1].biases_for_this_layer = np.reshape(self.global_best_position[lim3:lim4],(nn.hidden_units,1))
			lim5 = lim4 + np.size(nn.layers[2].biases_for_this_layer)
			nn.layers[2].biases_for_this_layer = np.reshape(self.global_best_position[lim4:lim5],(nn.hidden_units,1))
			lim6 = lim5 + np.size(nn.layers[3].biases_for_this_layer)
			nn.layers[3].biases_for_this_layer = np.reshape(self.global_best_position[lim5:lim6],(nn.output_units, 1))

			nn.forward_propagation(self.Xv)
			validation_loss[i] = nn.calculate_network_loss(self.yv)
		
			
			self.w = self.w * self.wdamp
			if (i+1)%100==0:
				print("Iteration Number: ", i+1, "Best Cost: ", np_loss[i])

		
		return nn, np_loss, validation_loss
