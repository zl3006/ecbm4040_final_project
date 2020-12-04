import tensorflow as tf
from tf.keras import model
import math
import sys
import time

if __name__ == '__main__':

    # GPU settings
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    
    # loading dataset
    # train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    print('#training images = %d' % train_count)

    model = ResNet50(parameters = [3, 4, 6, 3]) # use needed parameter
    batch_size = 256  # adjust batch size
    epoches = 30  # adjust epoches

    # define loss and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adadelta()  # choose needed optimizer

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    for epoch in range(epoches): 
        patience = 5   # adjust patience for early stop
        counter = 0    # patience counter
        best_acc = None
        total_steps = 0
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        
        for images, labels in train_dataset:
            total_steps += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     epoches,
                                                                                     total_steps,
                                                                                     math.ceil(train_count / batch_size),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  epoches,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))
        if best_acc is None:  # save the first score and checkpoint
            best_acc = valid_accuracy.result()
        elif valid_accuracy.result() < best_acc: # check if the early stop condition meets
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {patience}')
            if self.counter >= self.patience:
                print("Early Stop at Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                      "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                          epoches,
                                                                          train_loss.result(),
                                                                          train_accuracy.result(),
                                                                          valid_loss.result(),
                                                                          valid_accuracy.result())))
                model.save_weights(filepath='models', save_format='tf') 
                break
        else: # if the loss improved again
            best_acc = valid_accuracy.result()
            counter = 0
    model.save_weights(filepath='models', save_format='tf') 




