#c:\py\tf\tf3.py
#Derivative
# Importing tensorflow version 1 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior() 
  
# Initializing placeholder variables of 
# the graph 
a = tf.placeholder(tf.float32) 
b = tf.placeholder(tf.float32) 
  
# Defining the operation 
c = tf.multiply(a, b) 
  
# Instantiating a tensorflow session 
with tf.Session() as sess: 
  
    # Computing the output of the graph by giving 
    # respective input values 
    #out = sess.run(tf.gradients(a, b), feed_dict={a: [15.0], b: [20.0]})[0][0] 
    #out = sess.run(tf.gradients(c, a), feed_dict={a: [15.0], b: [20.0]})[0][0] 
    #out = sess.run(c, feed_dict={a: [15.0], b: [20.0]})[0][0] 
    out = sess.run(c, feed_dict={a: 15.0, b: 20.0})#[0][0]  #OK
  
    # Computing the output gradient of the output with 
    # respect to the input 'a' 
    derivative_out_a = sess.run(tf.gradients(c, a), feed_dict={ 
                                a: [15.0], b: [20.0]})[0][0] 
  
    # Computing the output gradient of the output with 
    # respect to the input 'b' 
    derivative_out_b = sess.run(tf.gradients(c, b), feed_dict={ 
                                a: [15.0], b: [20.0]})[0][0] 
  
    # Displaying the outputs 
    print(f'c = {out}') 
    print(f'Derivative of c with respect to a = {derivative_out_a}') 
    print(f'Derivative of c with respect to b = {derivative_out_b}') 
