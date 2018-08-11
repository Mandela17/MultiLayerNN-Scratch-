from numpy import exp, array, random, dot
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# hm_lines = 100000
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
wrdss='ham'
ps = PorterStemmer()
learning_rate=0.10

class Preprocessing(object):
	def create_lexicon(self,pos):
		lexicon = []
		with open(pos,'r') as f:
			contents = f.readlines()
			for l in contents[:len(contents)]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)
		final_lexicon=[]
		for w in lexicon:
			if w not in stop_words:
				final_lexicon.append(w)
		w_counts = Counter(final_lexicon)
		final_lexicon_1 = [ps.stem(i) for i in final_lexicon]
		final_lexicon_2= [lemmatizer.lemmatize(i) for i in final_lexicon_1]
		w_counts = Counter(final_lexicon_2)
		l2 = []
		for w in w_counts:
			if w_counts[w] >=0:
			# if 5000 > w_counts[w] > 50:
				l2.append(w)
		return l2
	def sample_handling(self,sample,lexicon):
		featureset = []
		# print(len(lexicon))
		# y=[]
		y=np.zeros((5574,2),dtype=np.int)
		# print(yy)
		with open(sample,'r') as f:
			contents = f.readlines()
			i=0
			for l in contents[:len(contents)]:
				# print(l)
				current_words = word_tokenize(l.lower())
				current_words = [lemmatizer.lemmatize(i) for i in current_words]
				if wrdss in current_words:
					# y.append(1)
					y[i][0]=1
					y[i][1]=0
					i+=1
				else:
					# y.append(0)
					y[i][0]=0
					y[i][1]=1
					i+=1

				features = np.zeros(len(lexicon))
				for word in current_words:
					if word.lower() in lexicon:
						index_value = lexicon.index(word.lower())
						features[index_value] = 1
				features = list(features)
				featureset.append(features)
		# print(len(y))
		return featureset,y

class NeuralNetwork():
	def __init__(self):
		random.seed(1)

		# l2 is no_of_neurons in layer_2
		# l3 is no of_neuraons in layer_3
		# no of neurons in output_layer=2
		no_of_neurons_layer_2 = 100
		no_of_neurons_layer_3 = 50

		# Assigning the random weights 
		# Weight matrix size would be (no_of_nodes in previous layer) x (no_of_nodes in next layer)
		self.weights1 = 2 * random.random((8424, no_of_neurons_layer_2)) -1
		self.weights2 = 2 * random.random((no_of_neurons_layer_2, no_of_neurons_layer_3)) -1
		self.weights3 = 2 * random.random((no_of_neurons_layer_3, 2)) -1
		
	def __sigmoid(self, x):
		return 1/(1+np.exp(-x))

	def __sigmoid_derivative(self, x):
		return x*(1-x)
	def __softmax(self,x):
		exps = np.exp(x)
		return exps / np.sum(exps)
	def __softmax_derivative(self, x):
		return x*(1-x)


	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		cost=np.zeros(number_of_training_iterations)
		for iteration in range(number_of_training_iterations):
			for i in range(len(training_set_inputs)):
			# for i in range(10):
			# for i in range(len(training_set_inputs)):

			
				a2 = self.__sigmoid(dot(training_set_inputs[i], self.weights1))
				# print("a2")
				# print(a2)
				a3 = self.__sigmoid(dot(a2, self.weights2))
				# print("a3")
				# print(a3)
				output = self.__softmax(dot(a3, self.weights3))
				print("output")
				print(output)

				
				# squared_error=np.mean(np.square(training_set_outputs - output))
				squared_error=min(np.square(training_set_outputs[i] - output))
				# corect_logprobs = -np.log(training_set_outputs[i], output)
				# cost[iteration]=squared_error
				print("squared_error")
				print(squared_error)
				del4 = 2*(training_set_outputs[i] - output)*self.__softmax_derivative(output)
				# print("del4")
				# print(del4)

				del3 = dot(self.weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
				# print("del3")
				# print(del3)
				del2 = dot(self.weights2, del3)*(self.__sigmoid_derivative(a2).T)
				# print("del2")
				# print(del2)

				a3=np.matrix(a3)
				# print("a3 dimensions")
				# print(a3.shape[0],a3.shape[1])
				del4=np.matrix(del4)
				a2=np.matrix(a2)
				del3=np.matrix(del3)
				del2=np.matrix(del2)
				# print("del2_dim")
				# print(del2.shape)
				# print("del4 dim")
				# print(del4.shape[0],del4.shape[1])
				updates3 = dot(a3.T, del4)
				# print("updates3")
				# print(updates3)
				updates2 = dot(a2.T,del3)
				# print("updates2")
				# print(updates2)
				# print("gg")
				tts=np.matrix(training_set_inputs[i])
				# print("fghj")
				# print(tts.shape)
				updates1 = dot(tts.T, del2)
				# print("updates1")
				# print(updates1)

				#Updating weights
				self.weights1 += updates1*learning_rate
				self.weights2 += updates2*learning_rate
				self.weights3 += updates3*learning_rate
			cost[iteration]=squared_error
		return cost,self.weights1,self.weights2,self.weights3

	def forward_pass(self, inputs,w1,w2,w3):
		a2 = self.__sigmoid(dot(inputs,w1))
		# print("a2")
		# print(a2)
		a3 = self.__sigmoid(dot(a2,w2))
		# print("a3")
		# print(a3)
		output = self.__sigmoid(dot(a3,w3))
		for i in range(1):
			if output[i]>output[i+1]:
				output[i]=1
				output[i+1]=0
			else:
				output[i+1]=1
				output[i]=0
		return output

	def test(self,weights1,weights2,weights3,testing_set_inputs, testing_set_outputs, number_of_testing_iterations):
		cost2=np.zeros(number_of_testing_iterations)
		for iteration in range(number_of_testing_iterations):
			for i in range(len(testing_set_inputs)):
			# for i in range(len(testing_set_inputs)):

				
				a2 = self.__sigmoid(dot(testing_set_inputs[i], self.weights1))
				# print("a2")
				# print(a2)
				a3 = self.__sigmoid(dot(a2, self.weights2))
				# print("a3")
				# print(a3)
				output = self.__softmax(dot(a3, self.weights3))
				# print("output")
				# print(output)
				for i in range(1):
					if output[i]>output[i+1]:
						output[i]=1
						output[i+1]=0
					else:
						output[i+1]=1
						output[i]=0

				print(output)
				# squared_error=np.mean(np.square(testing_set_outputs - output))
				squared_error=min(np.square(testing_set_outputs[i] - output))
				# corect_logprobs = -np.log(testing_set_outputs[i], output)
				# cost[iteration]=squared_error
				print("squared_error")
				print(squared_error)
				del4 = 2*(testing_set_outputs[i] - output)*self.__softmax_derivative(output)
				# print("del4")
				# print(del4)

				del3 = dot(self.weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
				# print("del3")
				# print(del3)
				del2 = dot(self.weights2, del3)*(self.__sigmoid_derivative(a2).T)
				# print("del2")
				# print(del2)

				a3=np.matrix(a3)
				# print("a3 dimensions")
				# print(a3.shape[0],a3.shape[1])
				del4=np.matrix(del4)
				a2=np.matrix(a2)
				del3=np.matrix(del3)
				del2=np.matrix(del2)
				# print("del2_dim")
				# print(del2.shape)
				# print("del4 dim")
				# print(del4.shape[0],del4.shape[1])
				updates3 = dot(a3.T, del4)
				# print("updates3")
				# print(updates3)
				updates2 = dot(a2.T,del3)
				# print("updates2")
				# print(updates2)
				# print("gg")
				tts=np.matrix(testing_set_inputs[i])
				# print("fghj")
				# print(tts.shape)
				updates1 = dot(tts.T, del2)
				# print("updates1")
				# print(updates1)

				self.weights1 += updates1*learning_rate
				self.weights2 += updates2*learning_rate
				self.weights3 += updates3*learning_rate
			cost2[iteration]=squared_error
		return cost2



if __name__ == "__main__":
	neural_network = NeuralNetwork()
	PP = Preprocessing()
	lexicon=PP.create_lexicon("data.txt")
	tr_x,tr_y=PP.sample_handling("data.txt",lexicon)
	# print(tr_y)
	X=np.array(tr_x)
	# print (len(X))
	v=np.array(tr_y)
	# print(v)
	# np.random.shuffle(X)
	# np.random.shuffle(v)
	percentage=int(len(X)*0.80)
	train_x,test_x = X[:percentage,:],X[percentage:,:]
	train_y,test_y = v[:percentage,:],v[percentage:,:]
	# print(test_y)
	
	input_size_trx=train_x.shape[0]
	# print(input_size_trx)
	input_size_tsx=test_x.shape[1]

	
	xPredicted = train_x[1]
	# train_yy=train_y.reshape(-1,1)
	# test_yy=test_y.reshape(-1,1)
	


	print("############################## Trainning_begins ######################################")
	iterations=2
	cost1,w1,w2,w3=neural_network.train(train_x, train_y, iterations)
	np.savetxt("In_sample_Error_changes.txt", cost1, fmt="%s")
	np.savetxt("w1.txt", w1, fmt="%s")
	np.savetxt("w2.txt", w2, fmt="%s")
	np.savetxt("w3.txt", w3, fmt="%s")
	# print("ma")
	# print(w1)
	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(np.arange(iterations), cost1, 'r')  
	ax.set_xlabel('Iterations')  
	ax.set_ylabel('in-sample error')  
	ax.set_title('Sigmoid_soft_max in-sample error changes with Iterations in SGD')
	fig.savefig('Sigmoid_soft_max_in-sample error changes with Iterations in SGD.png')
	
	
	print("############################## Testing_begins ######################################")
	iterations_2=3
	cost2=neural_network.test(w1,w2,w3,test_x, test_y, iterations_2)
	np.savetxt("Out_of_sample_Error_changes.txt", cost2, fmt="%s")
	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(np.arange(iterations_2), cost2, 'r')  
	ax.set_xlabel('Iterations')  
	ax.set_ylabel('out-of-sample error')  
	ax.set_title('Sigmoid_soft_max out-of-sample error changes with Iterations in SGD')
	fig.savefig('Sigmoid_softmax_out-of-sample error changes with Iterations in SGD.png')
	

	correct = 0
	incorrect = 0 
	for i in range(len(X)):
		p=neural_network.forward_pass(X[i],w1,w2,w3)
		q=v[i]
		if p[0]==q[0] and p[1]==q[1]:
			correct+=1
		else:
			incorrect+=1
	print("correct",correct)
	print("incorrect",incorrect)
	print("Total_Message",len(X))
	print("Accuracy_1",(correct/len(X))*100)

	correct_2 = 0
	incorrect_2 = 0 
	for i in range(len(test_x)):
		p=neural_network.forward_pass(test_x[i],w1,w2,w3)
		q=test_y[i]
		if p[0]==q[0] and p[1]==q[1]:
			correct_2+=1
		else:
			incorrect_2+=1
	print("correct_2",correct_2)
	print("incorrect_2",incorrect_2)
	print("Total_Message",len(test_x))
	print("Accuracy_1",(correct_2/len(test_x))*100)

	'''
	print (neural_network.forward_propagation(array(train_x[0])),train_y[0])
	print (neural_network.forward_propagation(array(train_x[1])),train_y[1])
	print (neural_network.forward_propagation(array(train_x[2])),train_y[2])
	print (neural_network.forward_propagation(array(train_x[3])),train_y[3])
	print (neural_network.forward_propagation(array(train_x[4])),train_y[4])
	print (neural_network.forward_propagation(array(train_x[5])),train_y[5])
	print (neural_network.forward_propagation(array(train_x[6])),train_y[6])
	print (neural_network.forward_propagation(array(train_x[7])),train_y[7])
	print (neural_network.forward_propagation(array(train_x[8])),train_y[8])
	print (neural_network.forward_propagation(array(train_x[9])),train_y[9])
	print (neural_network.forward_propagation(array(train_x[10])),train_y[10])
	print (neural_network.forward_propagation(array(train_x[11])),train_y[11])
	print (neural_network.forward_propagation(array(train_x[12])),train_y[12])
	'''
	