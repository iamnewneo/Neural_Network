import numpy as np

class pcn:
	def __init__(self,inputs,targets):
		if np.ndim(inputs)>1:
			#number of inputs
			self.nIn = np.shape(inputs)[1]
		else:
			self.nIn = 1
		if np.ndim(targets)>1:
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

	def pcntrain(self,inputs,targets,eta,nIterations):
		#add one bias unit
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		change =range(self.nData)

		for i in range(nIterations):
			self.activations = self.pcnfwd(inputs)
			self.weights -=eta*np.dot(np.transpose(inputs),self.activations-targets)
			print("Iteration " + str(i))
			print("Current weights are :")
			print(self.weights)
			activations=self.pcnfwd(inputs)
			print("Outputs are")
			print(activations)

	def confmat(self,inputs,targets):
		#add the inputs that match bias input
		inputs=np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		outputs = np.dot(inputs,self.weights)

		nClasses = np.shape(targets)[1]
		if nClasses==1:
			nClasses=2
			outputs=np.where(outputs>0,1,0)
		else:
			#return index of maximum value in axis = 1 i.e index of max value in row
			outputs = np.argmax(outputs,1)
			targets = np.argmax(targets,1)
		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j]=np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print(cm)
		print(np.trace(cm)/np.sum(cm))
