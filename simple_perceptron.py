import numpy as nu

class pcn:
	def __init__(self,inputs,targets):
		if np.dim(inputs)>1:
			#number of inputs
			self.nIn = np.shape(inputs)[1]
		else:
			self.nIn = 1
		if np.dim(targets)>1:
			#number of outputs
			self.nOut = np.shape(targets)[1]
		else:
			self.nOut = 1

		
		#number of possible inputs
		self.nData=np.shape(inputs)[0]

		#intitialise weights randomly
		self.weights=np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

	def pcnfwd(self,inputs):
		#compute activations
		activations = np.dot(inputs,self.weights)
		#threshold of 0
		return np.where(activations>0,1,0)
