import tensorflow as tf
import os

def get_dataset_labels(path, data_file):
    data_file_path = os.path.join(path, data_file)
    if not os.path.exists(data_file_path):
        raise Exception(data_file_path, " is not exist")
    print("Processing %s, please wait ..." % data_file_path)
    import pickle
    with open(data_file_path, "rb") as file:
        data = pickle.load(file)
        return data['dataset'], data['labels']

train_dataset, train_labels = get_dataset_labels("./pickle", "train_pickle.pickle")
test_dataset,  test_labels  = get_dataset_labels("./pickle", "test_pickle.pickle")
print(train_dataset.shape)
print(test_dataset.shape)

num_valid = int(train_dataset.shape[0] * 0.3)
num_train = train_dataset.shape[0] - num_valid
batch_size = 64
img_height = train_dataset.shape[1]
img_width  = train_dataset.shape[2]
num_channels = 1 if 3 >= len(train_dataset.shape) else train_dataset.shape[3]
num_labels   = 11
num_digits = 5
patch_size = 5
num_depth = 64

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev= 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant( 0.1 ), shape=shape)

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, img_height, img_width, num_channels])
    tf_train_label   = [tf.placeholder(tf.float32, shape=[batch_size, num_labels], name="tf_train_labels_%d" % i) for i in range(num_digits)]

    w_conv1 = weight_variable([patch_size, patch_size, num_channels, num_depth])
    b_conv1 = bias_variable([num_depth])




