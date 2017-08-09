'''
Created on Ago 08, 2017

@author: Varela

For the class Data Science: Deep Learning convolutional neural networkds on theano and tensorflow
lecture #7; Using one dimensional convolution to apply echo to the hello world file
course url 1: https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
course url 2: https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
lecture url: https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/learn/v4/t/lecture/4847750?start=0

'''


import matplotlib.pyplot as plt 
import numpy as np 
import wave 
# import sys

from scipy.io.wavfile import write 

# If you right-click on the file and go to "Get Info", you can see:
# sampling rate = 16000 Hz
# bits per sample = 16
# The first is quantization in time
# The second is quantization in amplitude
# We also do this for images!
# 2^16 = 65536 is how many different sound levels we have
# 2^8 * 2^8 * 2^8 = 2^24 is how many different colors we can represent

spf = wave.open('helloworld.wav', 'r')


#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print 'numpy signal shape:', signal.shape 

plt.plot(signal)
plt.title("Hello world without echo")
plt.show()

delta  = np.array([1.0, 0., 0.])
noecho = np.convolve(signal, delta)
print "noecho signal:", noecho.shape 
assert(np.abs(noecho[:len(signal)]- signal).sum() < 1e-8)

noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)

filt    			= np.zeros(16000)
filt[0] 			= 1
filt[4000] 		= 0.6 
filt[8000] 		= 0.3 
filt[12000] 	= 0.2 
filt[15999] 	= 0.1

out = np.convolve(signal, filt)
out = out.astype(np.int16)
write('out.wav', 16000, out)

