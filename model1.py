import tensorflow as tf


# 神经网络模型
def mynn_inference(images, batch_size, n_classes):
    """
input:
images:图像矩阵，
batch_size:批次数量，
n_classes:类别数量
output: softmax_linear （未softmax的结果）
    """
    # 第一层卷积核 3*3 3通道 16个
    with tf.compat.v1.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        # 卷积层 strides = [1，x_movement,y_movement,1] padding valid/same
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name) # con1的命名空间

    with tf.compat.v1.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pooling1")
        # ksize 池化窗口大小，strides 步长
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1, alpha=0.001/9.0, beta=0.75, name="norm1")
        """
        sqr_sum[a, b, c, d] =sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
        output = input / (bias + alpha * sqr_sum) ** beta
        depth_radius:前多少个通道至后多少个通道(窗口半宽)
        bias：可选的float.默认为1，偏移
        alpha：可选的float.默认为1.比例因子,通常是正数.
        beta：可选的float.默认为0.5.指数.
        name：操作的名称(可选).
        """

    with tf.compat.v1.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        # 卷积层 strides = [1，x_movement,y_movement,1] padding valid/same
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.compat.v1.variable_scope("pooling2_lrn") as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1, alpha=0.001 / 9.0, beta=0.75, name="norm2")
        pool2 = tf.nn.max_pool2d(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pooling2")

    with tf.compat.v1.variable_scope("local3") as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1),)
        local3 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name=scope.name)

    with tf.compat.v1.variable_scope("local4") as scope:
        weights = tf.get_variable("weights",
                                  shape=[256, 512],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[512],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1),)
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name="local4")

    with tf.compat.v1.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[512, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1),)
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name="softmax_linear")
    return softmax_linear


def losses(logits, labels):
    """
    :param logits: 经过模型运算返回的tensor
    :param labels: 对应的标签
    :return: 损失函数
    """
    print(logits, labels)
    with tf.compat.v1.variable_scope("loss") as scope:
        # 把交叉熵和softmax放在一起是为了通过spares提高计算速度
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="loss_per_eg")
        loss = tf.reduce_mean(cross_entropy, name="loss")  # 求所有样本的平均loss
    return loss


def training(loss, learning_rate):
    """
    :param loss:损失函数
    :param learning_rate:学习率
    :return: 训练的最优值
    """
    with tf.name_scope("optimizer"):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    :param logits: 经过模型运算返回的tensor
    :param labels:
    :return: accuracy正确率
    """
    with tf.compat.v1.variable_scope("accuracy") as scope:
        prediction = tf.nn.softmax(logits)
        correct = tf.nn.in_top_k(prediction, labels, 1)
        # 这输出了一个batch_size bool数组, 如果目标类的预测是所有预测(例如i)中的前k个预测, 则条目out[i]为true
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
    return accuracy

