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
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
wrdss='ham'
ps = PorterStemmer()
Threshold=0.50
Learning_rate=0.10

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
		print("lan of:",len(l2))
		return l2

	def sample_handling(self,sample,lexicon):
		featureset = []
		y=[]
		with open(sample,'r') as f:
			contents = f.readlines()
			for l in contents[:len(contents)]:
				current_words = word_tokenize(l.lower())
				current_words = [lemmatizer.lemmatize(i) for i in current_words]
				if wrdss in current_words:
					y.append(1)
				else:
					y.append(0)

				features = np.zeros(len(lexicon))
				for word in current_words:
					if word.lower() in lexicon:
						index_value = lexicon.index(word.lower())
						features[index_value] = 1
				features = list(features)
				featureset.append(features)
		return featureset,y

class NeuralNetwork():
	def __init__(self):
		random.seed(1)

		# l2 is no_of_neurons in layer_2
		# l3 is no of_neuraons in layer_3
		# no of neurons in output_layer=1
		l2 = 100
		l3 = 50

		# Assigning the random weights 
		# Weight matrix size would be (no_of_nodes in previous layer) x (no_of_nodes in next layer)
		self.weights1 = 2 * random.random((8424, l2)) -1
		self.weights2 = 2 * random.random((l2, l3)) -1
		self.weights3 = 2 * random.random((l3, 1)) -1
		# print("synaptic")
		# print(self.weights1.shape[0],self.weights1.shape[1])
		# print("Synaptic3")
		# print(self.weights3)
	def __sigmoid(self, x):
		return 1/(1+exp(-x))

	# this the sigmoid function derivative required to calculate gradient
	def __sigmoid_derivative(self, x):
		return x*(1-x)

	# Train the neural network that means updates weight parameter
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		cost=np.zeros(number_of_training_iterations)
		for iteration in range(number_of_training_iterations):
			for i in range(len(training_set_inputs)):
			# for i in range(len(training_set_inputs)):

				
				a2 = self.__sigmoid(dot(training_set_inputs[i], self.weights1))
				# print("a2")
				# print(a2)
				a3 = self.__sigmoid(dot(a2, self.weights2))
				# print("a3")
				# print(a3)
				output = self.__sigmoid(dot(a3, self.weights3))
				# print("output")
				# print(output)

				
				# squared_error=np.mean(np.square(training_set_outputs - output))
				squared_error=np.square(training_set_outputs[i] - output)
				# cost[iteration]=squared_error
				print("squared_error")
				print(squared_error)
				del4 = 2*((training_set_outputs[i] - output))*self.__sigmoid_derivative(output)
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
				updates_3 = dot(a3.T, del4)
				# print("updates_3")
				# print(updates_3)
				updates_2 = dot(a2.T,del3)
				# print("updates_2")
				# print(updates_2)
				# print("gg")
				tts=np.matrix(training_set_inputs[i])
				# print("fghj")
				# print(tts.shape)
				updates_1 = dot(tts.T, del2)
				# print("updates_1")
				# print(updates_1)

				
				self.weights1 += updates_1*Learning_rate
				self.weights2 += updates_2*Learning_rate
				self.weights3 += updates_3*Learning_rate
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
		if output>Threshold:
			output=1
		else:
			output=0 
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
				output = self.__sigmoid(dot(a3, self.weights3))
				# print("output")
				# print(output)
				if output>Threshold:
					output=1
				else:
					output=0

				
				# squared_error=np.mean(np.square(testing_set_outputs - output))
				squared_error=np.square(testing_set_outputs[i] - output)
				# cost2[iteration]=squared_error
				print("squared_error")
				print(squared_error)
				del4 = 2*((testing_set_outputs[i] - output))*self.__sigmoid_derivative(output)
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
				updates_3 = dot(a3.T, del4)
				# print("updates_3")
				# print(updates_3)
				updates_2 = dot(a2.T,del3)
				# print("updates_2")
				# print(updates_2)
				# print("gg")
				tts=np.matrix(testing_set_inputs[i])
				# print("fghj")
				# print(tts.shape)
				updates_1 = dot(tts.T, del2)
				# print("updates_1")
				# print(updates_1)

				# Updating the weights
				self.weights1 += updates_1*Learning_rate
				self.weights2 += updates_2*Learning_rate
				self.weights3 += updates_3*Learning_rate
			cost2[iteration]=squared_error
		return cost2



if __name__ == "__main__":
	neural_network = NeuralNetwork()
	PP = Preprocessing()
	lexicon=PP.create_lexicon("data.txt")
	tr_x,tr_y=PP.sample_handling("data.txt",lexicon)
	X=np.array(tr_x)
	print ("vector_len",X.shape[1])
	v=np.array(tr_y)
	print ("output_len",len(v))
	# cnt=0
	# for i in range(len(v)):
	# 	if v[i]==1:
	# 		cnt+=1
	# print(cnt)

	# np.random.shuffle(X)
	# np.random.shuffle(v)
	percentage=int(len(X)*0.80)
	train_x,test_x = X[:percentage,:],X[percentage:,:]
	train_y,test_y = v[:percentage],v[percentage:]
	
	input_size_trx=train_x.shape[0]
	# print(input_size_trx)
	input_size_tsx=test_x.shape[1]

	
	xPredicted = train_x[1]
	train_yy=train_y.reshape(-1,1)
	test_yy=test_y.reshape(-1,1)
	


	
	
	print("#####################Training Begins Here######################")
	iterations=3
	cost1,w1,w2,w3=neural_network.train(train_x, train_yy, iterations)
	# print("cost1")
	# print (cost1)
	# print("ma")
	# print(w1)
	np.savetxt("In_sample_Error_changes.txt", cost1, fmt="%s")
	np.savetxt("w1.txt", w1, fmt="%s")
	np.savetxt("w2.txt", w2, fmt="%s")
	np.savetxt("w3.txt", w3, fmt="%s")

	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(np.arange(iterations), cost1, 'r')  
	ax.set_xlabel('Iterations')  
	ax.set_ylabel('in-sample error')  
	ax.set_title('Sigmoid_in-sample error changes with Iterations in SGD')
	fig.savefig('Sigmoid_in-sample error changes with Iterations in SGD.png') 
	
	print("##################Testing Begins Here##################")
	
	iterations_2=3
	cost3=neural_network.test(w1,w2,w3,test_x, test_yy, iterations_2)
	np.savetxt("Out_of_sample_Error_changes.txt", cost3, fmt="%s")
	# print("cost3")
	# print(cost3)
	fig, ax = plt.subplots(figsize=(12,8))  
	ax.plot(np.arange(iterations_2), cost3, 'r')  
	ax.set_xlabel('Iterations')  
	ax.set_ylabel('Out_of_sample_Error')  
	ax.set_title('Sigmoid_Out_of_sample_Error changes with Iterations in SGD')
	fig.savefig('Sigmoid_Out_of_sample_Error changes with Iterations in SGD.png') 
	
	
	correct=0
	incorrect=0
	for i in range(len(X)):
		predicted = neural_network.forward_pass(X[i],w1,w2,w3)
		actual = v[i]
		if predicted == actual:
			correct+=1
		else:
			incorrect+=1
	
	print("correct",correct)
	print("incorrect",incorrect)
	print("Total_Message",len(X))
	print("Accuracy_1",(correct/len(X))*100)

	correct_2=0
	incorrect_2=0
	for i in range(len(test_x)):
		predicted_2 = neural_network.forward_pass(test_x[i],w1,w2,w3)
		actual_2 = test_y[i]
		if predicted_2 == actual_2:
			correct_2+=1
		else:
			incorrect_2+=1
	
	print("correct_1",correct_2)
	print("incorrect_1",incorrect_2)
	print("Total_Message",len(test_x))
	print("Accuracy_2",(correct_2/len(test_x))*100)

	
	# print (neural_network.forward_pass(train_x[55],w1,w2,w3),train_y[55])
	# print (neural_network.forward_propagation(array(train_x[1])),train_y[1])
	# print (neural_network.forward_propagation(array(train_x[2])),train_y[2])
	# print (neural_network.forward_propagation(array(train_x[3])),train_y[3])
	# print (neural_network.forward_propagation(array(train_x[4])),train_y[4])
	# print (neural_network.forward_propagation(array(train_x[5])),train_y[5])
	# print (neural_network.forward_propagation(array(train_x[6])),train_y[6])
	# print (neural_network.forward_propagation(array(train_x[7])),train_y[7])
	# print (neural_network.forward_propagation(array(train_x[8])),train_y[8])
	# print (neural_network.forward_propagation(array(train_x[9])),train_y[9])
	# print (neural_network.forward_propagation(array(train_x[10])),train_y[10])
	# print (neural_network.forward_propagation(array(train_x[11])),train_y[11])
	# print (neural_network.forward_propagation(array(train_x[12])),train_y[12]) 
	

	
	