import tensorflow as tf
import numpy as np
import os
import glob
from tqdm import tqdm

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim
import cv2
from test import *

batch_size = 5 
num_epochs = 10
image_height = 720
image_width = 1280
image_size = 224
rnn_size = 256
num_classes = 5
learning_rate = 1e-3
timestamps = 32

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

means = [_R_MEAN, _G_MEAN, _B_MEAN]
def pre_process(images):
  processed_images = []
  for n in range(images.shape[0]):
    image = images[n]
    image = cv2.resize(image,  (int(224 * (image_width * 1.0 / image_height)) + 1 , 224))
    image = image[:224, 87 : 87 + 224,:]
    image = np.array(image, dtype=np.float32)
    for i in range(3):
      image[:,:,i] -= means[i]
    processed_images.append(image)
  processed_images = np.array(processed_images, dtype=np.float32)
  return processed_images

from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.control_flow_ops import with_dependencies

# If scopes is not an empty list, then only batch norms for that
# scope are updated.
def add_update_ops_bn(inputs, scopes = []):
  training_mode = True 
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  if update_ops:
    updates = tf.group(*update_ops)
    inputs = smart_cond(
        training_mode,
        lambda: with_dependencies([updates], inputs),
        lambda: inputs)
  return inputs

graph = tf.Graph()
with graph.as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    images = tf.placeholder(dtype=tf.float32, name="images", shape=[batch_size * timestamps, image_size, image_size, 3])
    targets = tf.placeholder(dtype=tf.int32, name="targets", shape=[batch_size,])

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=0.00001)):
        pre_logits, _ = resnet_v2.resnet_v2_50(images,
                                            num_classes=None,
                                            is_training=True)

    pre_logits = tf.stop_gradient(pre_logits)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=["resnet_v2_50/logits", "resnet_v2_50/AuxLogits"])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn("resnet_v2_50.ckpt", variables_to_restore)

    lm_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(rnn_size)])
    # Use the dynamic_rnn function of Tensorflow to run the embedded inputs
    # using the lm_cell you've created, and obtain the outputs of the RNN cell.
    # You have created a cell, which represents a single block (column) of the RNN.
    # dynamic_rnn will "copy" the cell for each element in your sequence, runs the input you provide through the cell,
    # and returns the outputs and the states of the cell.
    print "output of resnet = ", pre_logits
    pre_logits_per_video = tf.split(pre_logits, batch_size, axis = 0 )
    print "logits per videos = ", pre_logits_per_video
    feat_videos = []
    for item in pre_logits_per_video:
       feat_videos.append(tf.expand_dims(item, axis = 0))
    print "pre concat = ", feat_videos
    feat_videos = tf.squeeze(tf.concat(feat_videos, axis = 0))
    print "input to lstm = ", feat_videos

    outputs, states = tf.nn.dynamic_rnn(cell = lm_cell, inputs = feat_videos, dtype = tf.float32)

    # Use a dense layer to project the outputs of the RNN cell into the size of the
    # vocabulary (vocab_size).
    # output_logits should be of shape [None,input_length,vocab_size]
    # You can look at the tf.layers.dense function
    output_logits = tf.layers.dense(outputs[:,-1], units = num_classes)
    output_logits = add_update_ops_bn(output_logits)

    # Setup the loss: using the sparse_softmax_cross_entropy.
    # The logits are the output_logits we've computed.
    # The targets are the gold labels we are trying to match
    # Don't forget to use the targets_mask we have, so your loss is not off,
    # And your model doesn't get rewarded for predicting PAD tokens
    # You might have to cast the masks into float32. Look at the tf.cast function.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = targets, logits = output_logits))
    prediction = tf.cast(tf.argmax(output_logits, axis = -1), dtype=tf.int32)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(prediction, targets), dtype=tf.float32))

    # Setup an optimizer (SGD, RMSProp, Adam), you can find a list under tf.train.*
    # And provide it with a start learning rate.
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(learning_rate, global_step, 5000, 0.96, staircase = True)
    #optimizer = tf.train.RMSPropOptimizer(lr)  
    optimizer = tf.train.AdamOptimizer(lr)

    # We create a train_op that requires the optimizer we've created to minimize the
    # loss we've defined.
    # look for the optimizer.minimize function, define what should be miniminzed.
    # You can provide it with the provide an optional global_step parameter as well that keeps of how many
    # Optimizations steps have been run.
    train_op = optimizer.minimize(loss)
    saver = tf.train.Saver()

names = {"biden": 0, "hillary":1, "justin":2, "pelosi":3, "trump":4}
import glob
def get_data_for_class(name):
  print("Loading data for : " , name)
  files = glob.glob("/home/kratarth/Downloads/cs282/data/dataset/" + name + "/tf/*.tfrecords")
  print(files)
  # Use only 1 file for now
  num, videos = get_number_of_records(files, 32)
  return num, videos


def load_data():
  data = []
  for name, label in names.iteritems():
    num_videos, videos = get_data_for_class(name)
    for video in videos:
      data.append((label, pre_process(video[:,:,:,:3])))
  return data


import random
with tf.Session(graph=graph) as sess:
    #Initializations
    sess.run(tf.global_variables_initializer())
    init_fn(sess)
    data = load_data()
    #Training
    for epoch in range(num_epochs):
        random.shuffle(data)
        total_videos = len(data)
        total_loss = 0
        total_acc = 0
        for iter_ in range(len(data) // batch_size):
            batch = data[iter_ * batch_size : (iter_+1) * batch_size]
            X = np.concatenate([_[1] for _ in batch], axis = 0)
            y = np.array([_[0] for _ in batch], dtype=np.int32)
            _, np_loss, acc = sess.run([train_op, loss, accuracy ], feed_dict = {images : X, targets : y})
            total_loss += np_loss
            total_acc += acc
        print(iter_ + 1, total_loss, total_acc)
        print("Loss %f, Acc %f "%(total_loss * 1.0 / float(iter_ + 1), total_acc * 1.0 / float(total_videos)))
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)

