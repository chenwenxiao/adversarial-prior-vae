import tensorflow as tf


v = tf.get_variable('v', dtype=tf.float32, initializer=1.)
w = tf.get_variable('w', dtype=tf.float32, initializer=0.)
x = tf.identity(v)

with tf.control_dependencies([x, w.assign(v)]):
    y = v.assign(2.)
    z = tf.identity(x)

with tf.Session() as session:
    session.run(v.initializer)
    session.run(w.initializer)

    print(session.run([w, x, y, z]))
