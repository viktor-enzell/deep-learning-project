import numpy as np
'''
File which some tools that will be useful for the implementation
'''

def ComputePerplexity(predictions, onehot_labels):
	'''
	Function that, given the predictions of a model
	and the associated labels, will return
	the perplexity of a probability model

	The perplixity computed here is the exponentiation
	of the entropy. Another possible perplixity is
	the exponentiation of the cross-entropy but in
	our case it is not necessary.

	The perplixity is computable in any base
	but 2 is the most suitable one
	'''
	entropy = -np.mean(np.log2(np.sum(predictions * onehot_labels, axis=0)))
	perplexity = 2**entropy

	return perplexity

def Softmax(o):
	P = np.exp(o) / np.sum(np.exp(o), axis=0)
	return P