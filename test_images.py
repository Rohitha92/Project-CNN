import numpy as np
import tensorflow as tf
import glob
from Dataset import *
from scipy import misc

imagePath = '/home/rohitha/Documents/RVM/test_images/*.jpg'
labelsFullPath = '/home/rohitha/Documents/RVM/Checkpoints/outlabels.txt'


graph = tf.Graph()
with graph.as_default():
    #training parameters
    batch_size = 64
    learning_rate= 0.01
    dropout = 0.5  # probability of dropout
    image_size = 28
    num_labels = 3
    num_channels = 1  # grayscale
    num_steps = 500

    tf_valid_dataset = tf.placeholder(tf.float32, shape=(1, image_size, image_size, num_channels))

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
    valid_prediction = tf.nn.softmax(network(tf_valid_dataset))

    init = tf.global_variables_initializer()
    answer = None
    files = glob.glob(imagePath)
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for picture in files:
            image_data= misc.imread(picture, flatten=True)
            new_saver=tf.train.import_meta_graph('Checkpoints/newckpt.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('Checkpoints/'))
            image_data = misc.imresize(image_data, [28, 28])
            image_data = np.expand_dims(np.expand_dims(image_data, 0), -1)
            predictions = sess.run(valid_prediction, feed_dict={tf_valid_dataset: image_data})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-2:][::-1]  # Getting top 2 predictions
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]
            for node_id in top_k:
                score = predictions[node_id]
            #print('%s (score = %.5f)' % (human_string, score))
            answer = labels[top_k[0]]
            print answer

