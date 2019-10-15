import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import input_data
import model

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

train, train_label = input_data.get_file(train_dir)
train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

train_logits = model.mynn_inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.training(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

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
    train_writer = tf.compat.v1.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.compat.v1.train.Saver()
    #  队列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        for step in np.arange(MAX_STEP):
            print(step)
            if coord.should_stop():
                print("队列停止,step = %d" %step)
                break
            _op, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

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
        print("跳出循环，%d" % step)
        # ax.plot(step_list, cnn_list1)
        # bx.plot(step_list, cnn_list2)
        # plt.show()
    except tf.errors.OutOfRangeError:
        print("Done training, -- epoch limit reached")
    finally:
        coord.request_stop()

