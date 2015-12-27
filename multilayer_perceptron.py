import numpy as np

class mlp:

	def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
		#setup network
		self.nin = inputs.shape[1]
		self.nout = targets.shape[1]
		self.ndata = inputs.shape[0]
		self.nhidden = nhidden

		self.beta=beta
		self.momentum=momentum
		self.outtype=outtype

		#initialise network
		self.weights1=(np.random.rand(self.nin+1,self.nhidden))
		self.weights2=(np.random.rand(self.nhidden+1,self.nout))

	def mlptrain(self,inputs,targets,eta,niterations):
		#add the inputs that match the bias node
		inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
		change=range(self.ndata)
		updatew1=np.zeros((np.shape(self.weights1)))
		updatew2=np.zeros((np.shape(self.weights2)))

		for n in range(niterations):
			self.outputs=self.mlpfwd(inputs)
			error = 0.5*np.sum((self.outputs-targets)**2)
			if (np.mod(n,100)==0):
				print("Iteration : "+ str(n)+" Error : "+ str(error))

			#different types of output neurons
			if self.outtype=='linear':
				deltao=(self.outputs-targets)/self.ndata
			elif self.outtype=='logistic':
				deltao=self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
			elif self.outtype=='softmax':
				deltao=(self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata
			else:
				print("Error ivalid outtype")

			deltah=self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
			updatew1=eta*(np.dot(np.transpose(inputs),deltah[:,:-1]))+self.momentum*updatew1
			updatew2=eta*(np.dot(np.transpose(self.hidden),deltao))+self.momentum*updatew2
			self.weights1-=updatew1
			self.weights2-=updatew2

			#randomise order of inputs (not necesary for matrix based calculation)
			#np.random.shuffle(change)
			#inputs=inputs[change,:]
			#targets=targets[change,:]
	def mlpfwd(self,inputs):
		self.hidden=np.dot(inputs,self.weights1)
		self.hidden=1.0/(1.0+np.exp(-self.beta*self.hidden))
		self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)
		outputs = np.dot(self.hidden,self.weights2)
		#differnet types of output neurons
		if self.outtype=='linear':
			return outputs
		elif self.outtype=='logistic':
			return 1.0/(1.0+np.exp(-self.beta*outputs))
		elif self.outtype =='softmax':
			normalisers=np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
			return np.transpose(np.transpose(np.exp(outputs))/normalisers)
		else:
			print("Invalid outtype")

	def confmat(self,inputs,targets):
		inputs=np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
		outputs=self.mlpfwd(inputs)
		nclasses=np.shape(targets)[1]
		if nclasses==1:
			nclasses=2
			outputs=np.where(outputs>0.5,1,0)
		else:
			#1-of N encoding
			outputs=np.argmax(outputs,1)
			targets=np.argmax(targets,1)
		cm=np.zeros((nclasses,nclasses))
		for i in range(nclasses):
			for j in range(nclasses):
				cm[i,j]=np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
		print("Confusion matrix")
		print(cm)
		#np.trace gives sum about diagnol and np.sum gives sum of the matrix elements
		print("Percentage correct "+str((np.trace(cm)	/np.sum(cm))*100))