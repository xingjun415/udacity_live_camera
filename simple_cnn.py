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
kernel_size = 3
num_depth = 64
num_hidden = 256

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev= 0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant( 0.1 ), shape=shape)

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, img_height, img_width, num_channels])
    tf_train_label   = [tf.placeholder(tf.float32, shape=[batch_size, num_labels], name="tf_train_labels_%d" % i) for i in range(num_digits)]

    layer1_w = weight_variable([kernel_size, kernel_size, num_channels, num_depth])
    layer1_b = bias_variable([num_depth])

    layer2_w = weight_variable([kernel_size, kernel_size, num_depth, num_depth])
    layer2_b = bias_variable([num_depth])

    layer3_w = weight_variable([img_width // 4 * img_height // 4 * num_depth, num_hidden])
    layer3_b = bias_variable([num_hidden])

    layer4_w = weight_variable([num_hidden, num_digits * num_labels])
    layer4_b = bias_variable([num_digits * num_labels])

    def model(data):
        conv = tf.nn.conv2d(data, layer1_w, [1,2,2,1], padding="SAME")
        hidden = tf.nn.relu(conv + layer1_b)
        conv = tf.nn.conv2d(hidden, layer2_w, [1,2,2,1], padding="SAME")
        hidden = tf.nn.relu( conv + layer2_b)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape( hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_w)) + layer3_b
        output = tf.matmul(hidden, layer4_w) + layer4_b
        split_logits = tf.split( output, num_digits, 1)
        return split_logits

    logits = model(tf_train_dataset)
    loss = tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits(tf_train_label[i], logits[i]) for i in range(num_digits)], name="loss")

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay( learning_rate,global_step, 100000,
                                                learning_decay, name= "learning_rate")