import tensorflow as tf
import input_data

class MNIST:
  def __init__(self):
    return

  def init_params(self, dir, x_pixels, y_pixels, classify_numbers, learning_rate):
    '''
    :param dir: mnist images directory
    :param x_pixels: one image x pixels
    :param y_pixels: one image y pixels
    :param classify_numbers: classifying numbers
    :param learning_rate: set training model regression rate
    '''
    # buffer mnist image datas
    self.mnist = input_data.read_data_sets(dir, one_hot=True)
    # create regression model
    imag_pixels = x_pixels * y_pixels
    self.x = tf.placeholder("float", [None, imag_pixels])
    self.W = tf.Variable(tf.zeros([imag_pixels, classify_numbers]))
    self.b = tf.Variable(tf.zeros([classify_numbers]))
    self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
    # training model
    self.y_ = tf.placeholder("float", [None,classify_numbers])
    cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
    self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # init all variables
    self.init = tf.initialize_all_variables()
    # create session
    self.sess = tf.Session()

  def run_learning_model(self):
    self.sess.run(self.init)

    for i in range(1000):
      batch_xs, batch_ys = self.mnist.train.next_batch(100)
      self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

  def mnist_evaluate(self):
    # evaluate my model
    correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print self.sess.run(accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels})

if __name__ == "__main__":
  ministInstance = MNIST()
  ministInstance.init_params("MNIST_data/", 28, 28, 10, 0.01)
  ministInstance.run_learning_model()
  ministInstance.mnist_evaluate()