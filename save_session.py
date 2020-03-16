import tensorflow as tf

import os
dir = os.path.dirname(os.path.realpath(__file__))

# Currently, we are in the default graph scope

# Let's design some variables
v1 = tf.Variable(1. , name="v1")
v2 = tf.Variable(2. , name="v2")
# Let's design an operation
a = tf.add(v1, v2)

# We can check easily that we are indeed in the default graph
print(a.graph == tf.get_default_graph())
# -> True

# Let's create a Saver object
# By default, the Saver handles every Variables related to the default graph
all_saver = tf.train.Saver() 
# But you can precise which vars you want to save (as a list) and under which name (with a dict)
v2_saver = tf.train.Saver({"v2": v2}) 


# By default the Session handles the default graph and all its included variables
with tf.Session() as sess:
  # Init v1 and v2   
  sess.run(tf.global_variables_initializer())
  # Now v1 holds the value 1.0 and v2 holds the value 2.0
  # and we can save them
  all_saver.save(sess, dir + '/ABC/data-all')
  # or saves only v2