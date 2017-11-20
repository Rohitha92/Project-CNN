import numpy as np

# Adapted from the TensorFlow tutorial at
# https://www.tensorflow.org/versions/master/tutorials/index.html
class DataSet(object):
  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
      "images.shape: %s labels.shape: %s" % (images.shape,
                                             labels.shape))
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_data_sets(train_images, train_labels, validation_images, validation_labels, num_classes):
      class DataSets(object):
          pass

      data_sets = DataSets()
      data_sets.train = DataSet(train_images, dense_to_one_hot(train_labels, num_classes))
      data_sets.validation = DataSet(validation_images, dense_to_one_hot(validation_labels, num_classes))
      return data_sets