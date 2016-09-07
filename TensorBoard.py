
####### From TF graph, decide which node you want to annotate

# with tf.name_scope("test") as scope:
#tf.histogram_summary("weights",W), tf.scalar_summary("accuracy", accuracy)




####### Merge all summaries

#merged = tf.merge_all_summaries()





####### Create writer

#writer = tf.train.SummaryWriter("/tmp/mnist_logs",sess.graph_def)





####### Run summary merge and add_summary

# summary = sess.run()merged,...); writer.add_summary(summary);





####### Launch Tensorboard

# tensorboard --logdir=/tmp/mnist_logs

