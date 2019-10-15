import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import input_data1
import model1

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.0001
train_dir= 'E:\Jupyter\catanddog\ALLPetImages'
#  train_dir = 'E:\PyCharmProject\mycatvsdog\PetImages'
logs_train_dir = 'E:\PyCharmProject\mycatvsdog\log'

train, train_label = input_data1.get_file(train_dir)
dataset = input_data1.get_batch(train, train_label, BATCH_SIZE)
# iterator = dataset.make_initializable_iterator()
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

train_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_H, IMG_W, 3])
train_label_batch = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

train_logits = model1.mynn_inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model1.losses(train_logits, train_label_batch)
train_op = model1.training(train_loss, learning_rate)
train_acc = model1.evaluation(train_logits, train_label_batch)

summary_op = tf.compat.v1.summary.merge_all()

#  折线图
step_list = list(range(100))
cnn_list1 = []
cnn_list2 = []
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.yaxis.grid(True)
ax.set_title('accuracy', fontsize=14, y=1.02)
ax.set_xlabel("step")
ax.set_ylabel("accuracy")
bx = fig.add_subplot(1, 2, 2)
ax.yaxis.grid(True)
ax.set_title('loss', fontsize=14, y=1.02)
ax.set_xlabel("step")
ax.set_ylabel("loss")


with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # sess.run(iterator.initializer)
    train_writer = tf.compat.v1.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.compat.v1.train.Saver()

    next_image, next_label = sess.run(next_element)
    print(next_image.shape, next_label)
    try:
        for step in np.arange(MAX_STEP):
            next_image, next_label = sess.run(next_element)
            next_label = tf.cast(next_label, tf.int32)
            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc], feed_dict={train_batch: next_image, train_label_batch: next_label})
            if step % 2 == 0:
                print("Step %d ,train_loss = %.2f, train_accuracy = %.2f%%" % (step, tra_loss, tra_acc*100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)

            if step % 100 == 0:
                cnn_list1.append(tra_acc)
                cnn_list2.append(tra_loss)

            if step % 5000 == 0:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done training, -- epoch limit reached")
    print("跳出循环，%d" % step)
    ax.plot(step_list, cnn_list1)
    bx.plot(step_list, cnn_list2)
    plt.show()
