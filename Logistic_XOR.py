import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_txt/xor_train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# Placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# Our hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h)) # sigmoid

#cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# Minimize
a = tf.Variable(0.01) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the varibales. We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
with tf.Session() as sess:
    sess.run(init)

    # Fit the line.
    for step in range(1000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))


    #--------------- After Learning ---------------#

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict = {X:x_data, Y:y_data}))
    print("Accuracy :", accuracy.eval({X:x_data, Y:y_data}))

    ##Low accuracy compare with Deep