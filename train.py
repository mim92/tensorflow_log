import tensorflow as tf
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging

from model import CNN


def set_logger(log_file_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_file_name))


def train(sess, model, x_train, y_train, batch_size):
    # Use tqdm for progress bar
    t = tqdm(range(len(x_train) // batch_size))
    acces, losses = [], []
    for i in t:
        batch_x = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]
        feed_dict = {model.x: batch_x, model.y: batch_y}
        _, loss, acc = sess.run([model.train_step, model.cross_entropy_loss, model.accuracy], feed_dict)
        # print(loss, acc)
        losses.append(loss)
        acces.append(acc)
        t.set_postfix(train_loss=sum(losses) / len(losses), train_acc=sum(acces) / len(acces))
    return sum(losses) / len(losses), sum(acces) / len(acces)


def eval(sess, model, x_test, y_test, batch_size):
    acces, losses = [], []
    for i in range(len(x_test) // batch_size):
        batch_x = x_test[i * batch_size:(i + 1) * batch_size]
        batch_y = y_test[i * batch_size:(i + 1) * batch_size]
        feed_dict = {model.x: batch_x, model.y: batch_y}
        loss, acc = sess.run([model.cross_entropy_loss, model.accuracy], feed_dict)
        # print(loss, acc)
        losses.append(loss)
        acces.append(acc)
    return sum(losses) / len(losses), sum(acces) / len(acces)


def main():
    parser = argparse.ArgumentParser(description='train the model for all model')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_dir', type=str, default='training_model')

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    model_dir = args.model_dir

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = np.expand_dims(x_train, axis=-1) / 255., np.expand_dims(x_test, axis=-1) / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    # x_train = x_train[:1024]
    # x_test = x_test[:256]

    model = CNN()
    os.makedirs(model_dir, exist_ok=True)
    set_logger(os.path.join(model_dir, 'train.log'))
    save_dir = os.path.join(model_dir, 'weights')
    save_path = os.path.join(save_dir, 'epoch')
    begin_at_epoch = 0

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)  # will keep last 5 epochs
        sess.run(tf.global_variables_initializer())
        if os.path.isdir(save_dir):
            restore_from = tf.train.latest_checkpoint(save_dir)
            begin_at_epoch = int(restore_from.split('-')[-1])
            saver.restore(sess, restore_from)
            epochs += begin_at_epoch

        for epoch in range(begin_at_epoch, epochs):
            logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
            train_loss, train_acc = train(sess, model, x_train, y_train, batch_size)
            valid_loss, valid_acc = eval(sess, model, x_test, y_test, batch_size)
            logging.info('train/acc: {:.4f}, train/loss: {:.4f}'.format(train_acc, train_loss))
            logging.info('valid/acc: {:.4f}, valid/loss: {:.4f}'.format(valid_acc, valid_loss))
            saver.save(sess, save_path, global_step=epoch + 1)


if __name__ == '__main__':
    main()