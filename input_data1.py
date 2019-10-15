import os,random
import numpy as np
import tensorflow as tf

# input:文件夹地址 output:文件名列表，标签列表
'''
从文件导入
    从文件导入记录的典型管道有以下几个阶段：
        文件名列表
        可选文件名洗牌
        可选时期限制
        文件名队列
        用于文件格式的读取器
        读者用于读取记录的解码器
        可选预处理
        示例队列
'''


def get_file(file_dir):
    image_name = []
    labels = []
    category_list = []
    category_label = []
    i = -1

    for category in os.listdir(file_dir):
        path = os.path.join(file_dir, category)
        category_list.append(category)
        i = i + 1
        category_label.append(i)
        for file in os.listdir(path):
            image_name.append(path + "\\" + file)
            labels.append(i)
        print("For now,there are %d samples.Now the category is %s" % (len(image_name), category))

    temp = np.array([image_name, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.reshape(image_decoded, [208, 208, 3])
    image_resized = tf.cast(image_resized, tf.float32)/255
    label = tf.cast(label, tf.int32)
    return image_resized, label


def get_batch(image, label, batchsize):
    # tensorflow中张量数据类型转换
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image, label))
    dataset = dataset.map(_parse_function)
    dataset.batch(batchsize)
#    image_batch, label_batch = tf.train.batch([image, label], batch_size=batchsize, num_threads=64, capacity=capacity)
    # tf.train.batch是一个tensor队列生成器，作用是按照给定的tensor顺序，把batch_size个tensor推送到文件队列，作为训练一个batch的数据，等待tensor出队执行计算。
    return dataset

