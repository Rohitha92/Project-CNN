from Dataset import *
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile



mat_contents = sio.loadmat('RVMtrainvaldata.mat')
labels = mat_contents['y']
labels = np.where(labels == 10,0, labels)
labels = labels.reshape((labels.shape[0], 1))
X_train, X_val, y_train, y_val = train_test_split(mat_contents['X'], labels, test_size=0.10)
print(X_train.shape)
print(X_val.shape)

graph = tf.Graph()
with graph.as_default():
    #training parameters
    batch_size = 64
    learning_rate= 0.001
    dropout = 0.8  # probability of dropout
    image_size = 28
    num_labels = 3
    num_channels = 1  # grayscale
    num_steps = 30000

    mydata = read_data_sets(X_train, y_train, X_val, y_val, num_labels)

    #tf graph input
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(mydata.validation.images.reshape(-1, 28, 28, 1))  # reshape dataset
    dropout          = tf.placeholder(tf.float32)

    #define cnn architecture blocks

    def conv2d(x, W, b, strides=1, pad='SAME'):
        x= tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding=pad)
        x= tf.nn.bias_add(x,b)
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')


    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    #define layers of the network with weights and biases

    weights ={
        'w1': weight_variable([3, 3, 1, 32]),
        'w2': weight_variable([5, 5, 32, 64]),
        'w22': weight_variable([2, 2, 64, 128]),
        'w3': weight_variable([2*2*128, 512]),
        'w4': weight_variable([512, num_labels])
    }

    biases={
        'b1': bias_variable([32]),
        'b2': bias_variable([64]),
        'b22': bias_variable([128]),
        'b3': bias_variable([512]),
        'b4': bias_variable([num_labels])
    }


    #defining the network model

    def network(data):
        h_conv1 = conv2d(data, weights['w1'], biases['b1']) #28x28
        h_pool1 = maxpool2d(h_conv1, k=2) #14x14
        h_conv2 = conv2d(h_pool1, weights['w2'], biases['b2']) #14x14
        h_pool2 = maxpool2d(h_conv2, k=2) #7x7
        h_conv3 = conv2d(h_pool2, weights['w22'], biases['b22'], strides=2, pad='SAME') #4x4
        h_pool3 = maxpool2d(h_conv3, k=2) #2x2
        #fully connected
        h_pool2_flat = tf.reshape(h_pool3, [-1, 2*2*128])
        h_fc1= tf.add(tf.matmul(h_pool2_flat, weights['w3']), biases['b3'])
        h_fc1= tf.nn.relu(h_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout)
        out = tf.add(tf.matmul(h_fc1_drop, weights['w4']), biases['b4'])
        return out

    logits = network(tf_train_dataset)
    train_prediction = tf.nn.softmax(logits)
    final_tensor = tf.nn.softmax(logits, name='final_result')
    tf.add_to_collection("final_tensor", final_tensor)
    valid_prediction = tf.nn.softmax(network(tf_valid_dataset))

    loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels= tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #optimizer= tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        for step in range(1, num_steps):
            batch_x, batch_y=  mydata.train.next_batch(batch_size)
            batch_x = batch_x.reshape(-1, 28, 28, 1)
            feed_dict = {tf_train_dataset: batch_x, tf_train_labels: batch_y, dropout: 0.8}
            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print ('Minibatch step %d : %f' % (step, l))
                print ('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_y))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(feed_dict={dropout: 0.8}),
                                                           mydata.validation.labels))

        saver.save(sess, 'Checkpoints/newckpt')
