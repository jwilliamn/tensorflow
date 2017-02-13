#### Title: Tensorflow for Deep Learning Research
#### Description: Operations

import tensorflow as tf 
import numpy as np


## Visualizing with tensorboard
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))

# Close the writer when you're done using it
writer.close()

# To lauch tensorboard use tensorboard --logdir="./graphs" in terminal


# Making tensorboard display the names of the ops (operations)
a = tf.constant([2, 2], name="a")
b = tf.constant([3, 6], name="b")
x = tf.add(a, b, name="add")

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))

writer.close()


## Constant types
print('Constant types')
# Constant of 1d tensor (vector)
a = tf.constant([2, 2], name="vector")

# Constant of 2x2 tensor (matrix)
b = tf.constant([[0, 1], [2, 3]], name="matrix")

x = tf.add(a, b, name="add")
y = tf.mul(a, b, name="mul")

with tf.Session() as sess:
	print(sess.run(a))
	print(sess.run(b))

	x, y = sess.run([x, y])
	print(x, y)


# Elements of a specific values
zeros = tf.zeros([2,3], tf.int32)

inputTensor = tf.constant([[0,1],[2,3],[4,5]])
zerosLike = tf.zeros_like(inputTensor)

ones = tf.ones([2, 3], tf.int32)
onesLike = tf.ones_like(inputTensor)

filled = tf.fill([2,3], 8)

with tf.Session() as sess:
	print(sess.run(zeros))
	print('zeros like')
	print(sess.run(zerosLike))
	print(sess.run(ones))
	print(sess.run(onesLike))
	print(sess.run(filled))


# Constants as sequences
seq = tf.linspace(10.0, 13.0, 4, name="linspace")
rang = tf.range(3, 18, 3, name="range")
limit = tf.range(5)

with tf.Session() as sess:
	print(sess.run(seq))
	print(sess.run(rang))
	print(sess.run(limit))

#for x in tf.range(4):
#	print('This is a test')  # Error: Tensor object is not iterable


## Math Operations
a = tf.constant([3, 6])
b = tf.constant([2, 2])

add = tf.add(a, b)
add_n = tf.add_n([a, b, b])  # Equivalent to a + b + b
mul = tf.mul(a, b)
#mat_mul = tf.matmul(a,b)
mat_mul = tf.matmul(tf.reshape(a,[1,2]),tf.reshape(b,[2,1]))

with tf.Session() as sess:
	print(sess.run(add))
	print(sess.run(add_n))
	print(sess.run(mul))
	#print(sess.run(mat_mul))  # Value error
	print(sess.run(mat_mul))


## TensorFlow data types
t0 = 19
t00 = tf.zeros_like(t0)

t1 = ['apple','peach','banana']
t11 = tf.zeros_like(t1)  # ?? [b'' b'' b'']

t2 = [[True, False, False],
	  [False, False, True],
	  [False, True, False]]  # Treated as a 2-d tensor or matrix
t21 = tf.zeros_like(t2)  # 2x2 tensor, all elements are False
t22 = tf.ones_like(t2)   # 2x2 tensor, all elements are True


with tf.Session() as sess:
	print(sess.run(t00))
	print(sess.run(t11))
	print(sess.run(t21))
	print(sess.run(t22))



## Numpy data types
# Tensorflow's data types are based on those of Numpy
print(tf.int32 == np.int32)

a = tf.ones([2, 2], np.float32)

# Print out the graph def
myConst = tf.constant([1.0, 2.0], name="MyConst")

with tf.Session() as sess:
	print(sess.graph.as_graph_def())

# Declare variables
# Create variable a with scalar value
a = tf.Variable(2, name="scalar")
b = tf.Variable([2,3], name="vector")

c = tf.Variable([[0, 1], [2, 3]], name="matrix")  # create variable c as 2x2 matrix
d = tf.Variable(tf.zeros([784, 10]))  # Create variable w as 784 x 10 tensor, filled with zero

# Initialize variables before using them
init = tf.global_variables_initializer()  # Initializing all variables at once
init_ab = tf.variables_initializer([a, b], name="init_ab")  # Initializing only a subset
w = tf.Variable(tf.zeros([784,10]))  	# Initializing a single variable

with tf.Session() as sess:
	#print(sess.run(a))
	#print(sess.run(init))
	sess.run(init_ab)
	sess.run(w.initializer)

	print(w.eval())

# Variable assign
w = tf.Variable(10)
w.assign(100)

with tf.Session() as sess:
	sess.run(w.initializer)
	print(w.eval())   # Prints 10 why? because w.assign(100) creates an op but needs to be run
					  # to take effect

# Variable assign and run
w = tf.Variable(10)
assign_op = w.assign(100)

with tf.Session() as sess:
	sess.run(assign_op)
	print(w.eval())
	

# Create a variable whose original value is 2
myVar =  tf.Variable(2, name="myVar")
myVarTimesTwo = myVar.assign(2*myVar)

with tf.Session() as sess:
	sess.run(myVar.initializer)
	sess.run(myVarTimesTwo)  # >> 4
	sess.run(myVarTimesTwo)  # >> 8
	sess.run(myVarTimesTwo)  # >> 16

	print(myVar.eval())


# Assign add() and sub()
myVar = tf.Variable(10)

with tf.Session() as sess:
	sess.run(myVar.initializer)
	sess.run(myVar.assign_add(10))  # Increment by 10
	sess.run(myVar.assign_sub(2))   # decrement by 2

	print(myVar.eval())

# Each session maintains its own copy of variable
w = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(w.initializer)
sess2.run(w.initializer)

print(sess1.run(w.assign_add(10)))
print(sess2.run(w.assign_sub(2)))

print(sess1.run(w.assign_add(100)))
print(sess2.run(w.assign_sub(50)))

sess1.close()
sess2.close()


# Use a variable to initialize another variable (U = W*2)
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(2 * W.initialized_value())

with tf.Session() as sess:
	sess.run(W.initializer)
	sess.run(U.initializer)

	print(W.eval())
	print(U.eval())


# InteractiveSession
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
## We can just use 'c.eval()' without specifying the context 'sess'
print(c.eval())
sess.close()


# Placeholders are valid op 
#Create a placeholder of type float 32-bit is a vector of 3 elem
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b # short for tf.add()

with tf.Session() as sess:
	# feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
	# fetch value of c
	print(sess.run(c, {a:[1, 2, 3]}))


# Feeding values to TF ops
# Create operations, tensor, etc (using the default graph)
a = tf.add(2, 5)
b = tf.mul(a, 3)

with tf.Session() as sess:
	# define the dictionary that says to replace the value of a with '15'
	replace_dict = {a: 15}
	# Run de session, passing in 'replace_dict' as the value o 'feed_dict'
	print(sess.run(b, feed_dict=replace_dict))
	#print(b.eval())

# Normal loading vs Lazy loading
## Normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)  # create the node for add node before executing the graph

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	for _ in range(10):
		sess.run(z)
	writer.close()

## Lazy loading  (Avoid it!!)
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	for _ in range(10):
		sess.run(tf.add(x, y))  # We save one line of code
	writer.close()