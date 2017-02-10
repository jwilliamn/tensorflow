#### Title: Tensorflow for Deep Learning Research
#### Description: General overview

import tensorflow as tf 


# Data flow graphs
a = tf.add(3, 5)

# Nodes are operators, variables, and constants
# Edges: tensors 
# tensors are data
print(a)

# To get the value of a create a session
sess = tf.Session()

print(sess.run(a))
sess.close()

# "with" clause takes care of sess.close()

with tf.Session() as sess:
	print(sess.run(a))


# More graphs
x = 2
y = 3

op1 = tf.add(x, y)
op2 = tf.mul(x, y)
useless = tf.mul(x, op1)
op3 = tf.pow(op2, op1)

with tf.Session() as sess:
	#op3 = sess.run(op3)
	#print(op3)

	op3, notUseless = sess.run([op3, useless])
	print(notUseless)


# Distributed computation

# creates a graph
#with tf.device('/gpu:2'):
#	a = tf.constants([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
#	b = tf.constants([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
#	c = tf.matmul(a, b)


# Create a graph
g = tf.Graph()

with g.as_default():
	a = 3
	b = 5
	#x = tf.add(3, 5)
	x = tf.add(a, b)

#sess = tf.Session(graph=g)
with tf.Session(graph=g) as sess:
	print(sess.run(x))


# To handle the default graph
g = tf.get_default_graph()
print(g)


# Multiple graphs
g1 = tf.get_default_graph()
g2 = tf.Graph()

# add ops to the default graph
with g1.as_default():
	a = tf.constant(3)

# add ops to the user created graphs
with g2.as_default():
	b = tf.constant(5)


print(a)
print(b)