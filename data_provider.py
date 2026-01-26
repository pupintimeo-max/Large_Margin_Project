import os
import numpy as np
import tensorflow as tf

# Global constants describing the MNIST data set.
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
NUM_TRAIN_EXAMPLES = 60000
NUM_TEST_EXAMPLES = 10000

LABEL_BYTES = 1
IMAGE_BYTES = IMAGE_SIZE ** 2 * NUM_CHANNELS


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.io.gfile.GFile(filename, "rb") as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError(
                f"Invalid magic number {magic} in MNIST file {f.name}"
            )
        if rows != 28 or cols != 28:
            raise ValueError(
                f"Invalid MNIST file {f.name}: Expected 28x28 images, found {rows}x{cols}"
            )


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.io.gfile.GFile(filename, "rb") as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError(
                f"Invalid magic number {magic} in MNIST file {f.name}"
            )


def get_image_label_from_record(image_record, label_record):
    """Decodes the image and label information from one data record."""
    # Convert from tf.string to tf.uint8.
    image = tf.io.decode_raw(image_record, tf.uint8)
    # Convert from tf.uint8 to tf.float32.
    image = tf.cast(image, tf.float32)

    # Reshape image to correct shape.
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    # Normalize from [0, 255] to [0.0, 1.0]
    image = image / 255.0

    # Convert from tf.string to tf.uint8.
    label = tf.io.decode_raw(label_record, tf.uint8)
    # Convert from tf.uint8 to tf.int32.
    label = tf.cast(label, tf.int32)
    # Reshape label to correct shape.
    label = tf.reshape(label, [])  # label is a scalar
    return image, label


def create_mnist_dataset(data_dir, subset, batch_size, is_training=False):
    """Creates a tf.data.Dataset for MNIST.

    Args:
        data_dir: Directory containing the MNIST data files.
        subset: Data subset 'train' or 'test'.
        batch_size: Batch size for the dataset.
        is_training: Whether this is for training (enables shuffling).

    Returns:
        A tf.data.Dataset yielding (images, labels) tuples.
    """
    if subset == "train":
        images_file = os.path.join(data_dir, "train-images.idx3-ubyte")
        labels_file = os.path.join(data_dir, "train-labels.idx1-ubyte")
        num_examples = NUM_TRAIN_EXAMPLES
    elif subset == "noisy":
        images_file = os.path.join(data_dir, "noisy_train-images-idx3-ubyte")
        labels_file = os.path.join(data_dir, "noisy_train-labels-idx1-ubyte")
        num_examples = NUM_TRAIN_EXAMPLES
    elif subset == "few":
        images_file = os.path.join(data_dir, "train-images.idx3-ubyte")
        labels_file = os.path.join(data_dir, "train-labels.idx1-ubyte")
        num_examples = 68
    elif subset == "test":
        images_file = os.path.join(data_dir, "t10k-images.idx3-ubyte")
        labels_file = os.path.join(data_dir, "t10k-labels.idx1-ubyte")
        num_examples = NUM_TEST_EXAMPLES
    else:
        raise ValueError(f"Invalid subset: {subset}")

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    # Construct fixed length record dataset.
    dataset_images = tf.data.FixedLengthRecordDataset(
        images_file, IMAGE_BYTES, header_bytes=16
    )
    dataset_labels = tf.data.FixedLengthRecordDataset(
        labels_file, LABEL_BYTES, header_bytes=8
    )

    dataset = tf.data.Dataset.zip((dataset_images, dataset_labels))

    dataset = dataset.take(num_examples)

    if is_training:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.repeat() if is_training else dataset
    dataset = dataset.map(
        get_image_label_from_record,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, num_examples, NUM_CLASSES


class MNISTDataset:
    """MNIST dataset wrapper for compatibility with existing code.

    Attributes:
        dataset: A tf.data.Dataset yielding (images, labels) tuples.
        num_examples: Number of examples in the dataset.
        num_classes: Number of classes (10 for MNIST).
        subset: Data subset name.
    """

    def __init__(self, data_dir, subset, batch_size, is_training=False):
        self.dataset, self.num_examples, self.num_classes = create_mnist_dataset(
            data_dir, subset, batch_size, is_training
        )
        self.subset = subset
        self.batch_size = batch_size