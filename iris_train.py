import numpy as np

def preprocessIris(infile,outfile):
	stext1 = 'Iris-setosa'
	stext2 = 'Iris-versicolor'
	stext3 = 'Iris-virginica'
	rtext1 = '0'
	rtext2 = '1'
	rtext3 = '2'

	fid = open(infile,'r')
	oid = open(outfile,'w')

	for s in fid:
		if s.find(stext1)>-1:
			oid.write(s.replace(stext1,rtext1))
		elif s.find(stext2)>-1:
			oid.write(s.replace(stext2,rtext2))
		elif s.find(stext3)>-1:
			oid.write(s.replace(stext3,rtext3))
	fid.close()
	oid.close()
preprocessIris('iris.data','iris_proc.data')
#normalising iris data by dividing my mx instead of dividing by variance
iris= np.loadtxt('iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),np.abs(iris.min(axis=0)*np.ones((1,5)))),axis=0).max(axis=0)
iris[:,:4]=iris[:,:4]/imax[:4]

#split into train,validation,and test sets
target=np.zeros((np.shape(iris)[0],3))
indices = np.where(iris[:,4]==0)
target[indices,0]=1
indices = np.where(iris[:,4]==1)
target[indices,1]=1
indices = np.where(iris[:,4]==2)
target[indices,2]=1

#randomly order data
#order = list(range(np.shape(iris)[0]))
#np.random.shuffle(order)
#iris=iris[order,:]
#target=target[order,:]
train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

#train the network
import multilayer_perceptron as mlp
net=mlp.mlp(train,traint,5,outtype='logistic')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)

#funttion to check the ouput of the single data and check its output
def check_element(inputs):
	inputs = np.concatenate((inputs,-np.ones((inputs.shape[0],1))),axis=1)
	outputs = net.mlpfwd(inputs)
	out = np.argmax(outputs,1)
	print("________PREDICTED VALUE__________")
	print(out)
print("_____________ORIGINAL VALUE")
print(iris[80,4])
check_element(iris[79:80,0:4])