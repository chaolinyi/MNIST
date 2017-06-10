import tensorflow as tf
import input_data

class MNIST:
  def __init__(self, dir):
    self.mnist = input_data.read_data_sets(dir, one_hot=True)

  def mnist_softmax(self, x_pixels, y_pixels, classify_numbers):
    imag_pixels = x_pixels * y_pixels
    # regression model
    self.x = tf.placeholder("float", [None, imag_pixels])
    self.W = tf.Variable(tf.zeros([imag_pixels,classify_numbers]))
    self.b = tf.Variable(tf.zeros([classify_numbers]))
    self.y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)

    # training model
    learning_rate = 0.01
    self.y_ = tf.placeholder("float", [None,classify_numbers])
    cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    self.sess = tf.Session()
    self.sess.run(init)

    for i in range(1000):
      batch_xs, batch_ys = self.mnist.train.next_batch(100)
      self.sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

  def mnist_evaluate(self):
    # evaluate my model
    correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print self.sess.run(accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels})

if __name__ == "__main__":
  ministInstance = MNIST("MNIST_data/")
  ministInstance.mnist_softmax(28, 28, 10)
  ministInstance.mnist_evaluate()