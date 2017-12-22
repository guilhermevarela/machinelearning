'''
	Created on Dec 22, 2017

	@author: Varela

	Tensorflow scan 0

	Using scan to square an array 
	ref: https://rdipietro.github.io/tensorflow-scan-examples/

'''
import numpy as np 
import tensorflow as tf 


def square(_,this_element):
	return this_element*this_element

def fibo(a,_):
	return (a[1], a[0]+a[1])

def cumsum(a,b):
	return a+b

def delay(a,b):
	return (a[1],b)

def ts(a,b):
	return (b, tf.log(b/a[0]))

def matmul(M, a, b):
	return tf.matmul(M, tf.transpose(b))

N=10
X_init=np.arange(N)
X=tf.placeholder(tf.int32, shape=(N,), name='X')
V=tf.placeholder(tf.float32, shape=(N,), name='V')
Z=tf.placeholder(tf.int32, shape=(N,N), name='Z')

Y=tf.Variable(X_init.astype(np.int32), name='Y') 



square_op=tf.scan(
	fn=square,
	elems=X,
	name='square_operation'
)

fibo_init=(np.array(0), np.array(1))
fibo_op=tf.scan(
	fn=fibo,
	elems=V,
	initializer=fibo_init,
	name='fibonacci_operation'
)


cumsum_op=tf.scan(
	fn=cumsum,
	elems=X,
	name='cumsum_operation'
)


delay_op=tf.scan(
	fn=delay,
	elems=X,
	initializer=(0, 0),
	name='delay_operation'
)

ts_op=tf.scan(
	fn=ts,
	elems=V,
	initializer=(np.array(1.0).astype(np.float32), np.array(0.0).astype(np.float32)),
	name='timeseries_operation'
)

M= np.tile(np.arange(N), (N,1)).astype(np.int32)

fn_matmul=lambda x,y : matmul(M,x,y)
matmul_op=tf.scan(
	fn=fn_matmul,
	elems=Z,	
	initializer=np.zeros((N,1),dtype=np.int32),
	name='matmul_operation'
)



init 				= tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)

	print("square_operation")
	Y= session.run(square_op, feed_dict={X: X_init})

	print(X_init)
	print(Y)

	print("fibo_operation")
	X_init=np.array([1]+[0]*(N-1), dtype=np.int32)
	F= session.run(fibo_op, feed_dict={V: X_init})

	print(X_init)
	print(F)


	print("cumsum_operation")
	X_init=np.arange(N)
	C= session.run(cumsum_op, feed_dict={X: X_init})

	print(X_init)
	print(C)

	print("delay_operation")
	X_init=np.arange(N)
	D= session.run(delay_op, feed_dict={X: X_init})

	print(X_init)
	print(D)

	print("ts_operation")
	X_init=np.arange(N)
	T= session.run(ts_op, feed_dict={V: X_init.astype(np.float32)})

	print(X_init)
	print(T)

	print("matmul_operation")
	X_init=np.eye(N)
	M= session.run(matmul_op, feed_dict={Z: np.eye(N)})

	print(X_init)
	print(T)


