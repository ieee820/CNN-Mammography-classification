# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Implement batch normalization
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import numpy

import ddsm

FLAGS = None


def train():
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir,
  #                                  one_hot=True,
  #                                  fake_data=FLAGS.fake_data)
  #dataset = ddsm.get_dataset()
  patch_shape = ddsm.getSize()
  train_set_size = ddsm.getDataTrainSize()
  batch_size = 16
  steps_per_epoch = int(numpy.ceil(train_set_size/batch_size))
  print('Steps per epoch', steps_per_epoch)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, patch_shape[0] * patch_shape[0]], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, patch_shape[0], patch_shape[1], 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    return
    """
    with tf.name_scope('summaries'):
      
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
      """


  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  def conv_layer(images, filter_dim, input_channels, output_channels, layer_name, act=tf.nn.relu):
    with tf.variable_scope(layer_name) as scope:
      kernel = weight_variable([filter_dim, filter_dim, input_channels, output_channels])
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = bias_variable([output_channels])
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = act(pre_activation, name=scope.name)
      with tf.name_scope('weights'):
        variable_summaries(kernel)
      return conv1

  xa = conv_layer(image_shaped_input, 3, 1, 32, 'conv1')
  #xa = tf.nn.max_pool(xa, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  
  #xa = conv_layer(xa, 3, 32, 46, 'conv2a')
  res1in = xa
  xa = conv_layer(xa, 3, 32, 32, 'conv2b')
  xa = conv_layer(xa, 3, 32, 32, 'conv2c') + res1in
  xa = tf.nn.max_pool(xa, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  xa = conv_layer(xa, 3, 32, 64, 'conv3a')
  res2in = xa
  xa = conv_layer(xa, 3, 64, 64, 'conv3b')
  xa = conv_layer(xa, 3, 64, 64, 'conv3c') + res2in
  xa = tf.nn.max_pool(xa, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  xa = conv_layer(xa, 3, 64, 128, 'conv4a')
  res3in = xa
  xa = conv_layer(xa, 3, 128, 128, 'conv4b')
  xa = conv_layer(xa, 3, 128, 128, 'conv4c') + res3in
  xa = tf.nn.max_pool(xa, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  xa = conv_layer(xa, 3, 128, 256, 'conv5a')
  res4in = xa
  xa = conv_layer(xa, 3, 256, 256, 'conv5b')
  xa = conv_layer(xa, 3, 256, 256, 'conv5c') + res4in
  xa = tf.nn.max_pool(xa, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

  xa = conv_layer(xa, 3, 256, 512, 'conv6a')
  res5in = xa
  xa = conv_layer(xa, 3, 512, 512, 'conv6b')
  xa = conv_layer(xa, 3, 512, 512, 'conv6c') + res5in

  fc = 4**2 * 512
  xa = tf.reshape(xa, [-1, fc])
  hidden1 = nn_layer(xa, fc, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    lrn_rate = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  #y = nn_layer(dropped, 500, 2, 'layer2', act=tf.identity)
  #y = nn_layer(y, fc, 2, 'layer2', act=tf.identity)
  y = nn_layer(xa, fc, 2, 'layer2', act=tf.identity)
  
  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
    #diff = tf.nn.weighted_cross_entropy_with_logits(y, y_, tf.constant([4.0, 1.0]))
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    #train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
    #    cross_entropy)
    #train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9, use_nesterov=True).minimize(
    #    cross_entropy)
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries
 
  def feed_dict(train):
    k = FLAGS.dropout
    if train:
      xs, ys = ddsm.getTrainBatch(batch_size)
      return {x: xs, y_: ys, keep_prob: k, lrn_rate: FLAGS.learning_rate}
    else:
      xs, ys = ddsm.getTest()
      return {x: xs, y_: ys, keep_prob: k, lrn_rate: FLAGS.learning_rate}

  
  #def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
  """  if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}
  #"""
  def update_progress(progress):
      #print("", end='\r')
      width = 50
      hashtags = int(progress*width)

      print('\r[%s] %.2f%%' % ('#'* hashtags + ' ' * (width - hashtags), progress*100), end="")
      sys.stdout.flush()
      #print('.', end="")

  for i in range(FLAGS.max_steps):
    if not i == 0 and i % steps_per_epoch == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('\nAccuracy at step %s: %s' % (i, acc))
      FLAGS.learning_rate *= 0.9
      print('Learning rate %f' % FLAGS.learning_rate)
    #else:  # Record train set summaries, and train
    if i % 10 == 0:  # Record execution stats
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
      #print('Adding run metadata for', i)
    else:  # Record a summary
      summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)
    if i % steps_per_epoch == 0:
      print('\nEpoch %d' % (i / steps_per_epoch))
    update_progress(((i) % steps_per_epoch) / (steps_per_epoch-1))
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
