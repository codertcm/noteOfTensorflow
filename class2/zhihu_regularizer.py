import tensorflow as tf

sess = tf.Session()
weight_decay = 0.1
tmp = tf.constant([0, 1, 2, 3], dtype=tf.float32)

l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
a = tf.get_variable("I_am_a", regularizer=l2_reg, initializer=tmp)

# 上面代码的等价代码
'''
a=tf.get_variable("I_am_a",initializer=tmp)
a2=tf.reduce_sum(a*a)*weight_decay/2;
tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,a2)
'''
sess.run(tf.global_variables_initializer())
keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
for key in keys:
    print("%s : %s" % (key.name, sess.run(key)))
print(sess.run(a))

l2_loss = loss + tf.add_n(keys)

# 输出：
# I_am_a/Regularizer/l2_regularizer:0 : 0.7
# [ 0.  1.  2.  3.]
