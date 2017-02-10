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
init = tf.global_variables_initializer()

with tf.Session() as sess:
	print(sess.run(int))
	#print(sess.run(a))
	