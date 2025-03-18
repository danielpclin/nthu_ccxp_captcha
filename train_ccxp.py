import os
import pathlib
import pickle

import wandb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical
from wandb.integration.keras import WandbCallback

from helpers import MinimumEpochEarlyStopping, MinimumEpochReduceLROnPlateau, get_dataset_dataframe, alphabet

# Setup mixed precision
mixed_precision.set_global_policy('mixed_float16')


def plot(history, version_num):
    fig = plt.figure(figsize=(20, 15))

    # Plot training accuracy
    plt.subplot(2, 1, 1)
    training_accuracy_keys = [key for key in history.keys() if 'accuracy' in key and 'val' not in key]
    for key in training_accuracy_keys:
        plt.plot(history[key])
    plt.title('Model training accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(training_accuracy_keys)
    plt.ylim(0.8, 1)
    plt.grid()

    # Plot training loss
    plt.subplot(2, 1, 2)
    training_loss_keys = [key for key in history.keys() if 'loss' in key and 'val' not in key]
    for key in training_loss_keys:
        plt.plot(history[key])
    plt.title('Model training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(training_loss_keys)
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/ccxp/train_{version_num}.png")
    plt.close(fig)

    fig = plt.figure(figsize=(20, 15))

    # Plot val accuracy
    plt.subplot(2, 1, 1)
    val_accuracy_keys = [key for key in history.keys() if 'accuracy' in key and 'val' in key]
    for key in val_accuracy_keys:
        plt.plot(history[key])
    plt.title('Model val accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(val_accuracy_keys)
    plt.ylim(0.8, 1)
    plt.grid()

    # Plot val loss
    plt.subplot(2, 1, 2)
    val_loss_keys = [key for key in history.keys() if 'loss' in key and 'val' in key]
    for key in val_loss_keys:
        plt.plot(history[key])
    plt.title('Model val loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(val_loss_keys)
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/ccxp/val_{version_num}.png")
    plt.close(fig)


def train(version_num, img_size=(104, 32), train_dir="dataset/ccxp/dataset", val_dir="dataset/ccxp/validate", batch_size=512):
    checkpoint_path = f'checkpoints/ccxp/{version_num}.hdf5'
    log_dir = f'logs/ccxp/{version_num}'
    epochs = 100
    learning_rate = 0.0001
    optimizer = Adam(learning_rate)
    img_width, img_height = img_size
    run = wandb.init(project="nthu_ccxp_captcha", entity="danielpclin", reinit=True, config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "version": version_num,
        "img_width": img_width,
        "img_height": img_height,
    })
    train_df = get_dataset_dataframe(train_dir, out=6)
    validate_df = get_dataset_dataframe(val_dir, out=6)
    train_data_gen = ImageDataGenerator(rescale=1. / 255)
    validate_data_gen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data_gen.flow_from_dataframe(dataframe=train_df, directory='.', x_col="file",
                                                         y_col=[f'label_{i}' for i in range(1, 7)],
                                                         class_mode="multi_output", target_size=(img_height, img_width),
                                                         batch_size=batch_size)
    validation_generator = validate_data_gen.flow_from_dataframe(dataframe=validate_df, directory='.', x_col="file",
                                                                 y_col=[f'label_{i}' for i in range(1, 7)],
                                                                 class_mode="multi_output",
                                                                 target_size=(img_height, img_width),
                                                                 batch_size=batch_size)
    input_shape = (img_height, img_width, 3)
    main_input = Input(shape=input_shape)
    x = main_input
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    out = [Dense(len(alphabet), name=f'label_{i}', activation='softmax')(x) for i in range(1, 7)]
    model = Model(main_input, out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')

    early_stop = MinimumEpochEarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto', min_epoch=15)
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduce_lr = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, cooldown=1, mode='auto',
                                              min_lr=0.00001, min_epoch=15)
    wandb_callback = WandbCallback()
    callbacks_list = [tensor_board, early_stop, checkpoint, reduce_lr, wandb_callback]

    model.summary()
    train_history = model.fit(
        train_generator,
        steps_per_epoch=np.ceil(train_generator.n // train_generator.batch_size),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=np.ceil(validation_generator.n // validation_generator.batch_size),
        verbose=1,
        callbacks=callbacks_list
    )
    os.makedirs(pathlib.Path("results/ccxp"), exist_ok=True)
    with open(f"results/ccxp/{version_num}.pickle", "wb") as file:
        pickle.dump(train_history.history, file)

    plot(train_history.history, version_num)

    run.finish()
    K.clear_session()


def main():
    with open('ccxp_version.txt', 'r+') as f:
        version_num = int(f.read())
        version_num += 1
        f.seek(0)
        f.write(str(version_num))
        f.truncate()
    train(version_num=version_num)


if __name__ == "__main__":
    main()
