import os
import pathlib
import pickle

import wandb
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical
from wandb.integration.keras import WandbCallback

# Setup mixed precision
mixed_precision.set_global_policy('mixed_float16')
alphabet = list('0123456789')
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


# Define residual blocks
def Conv2D_BN_Activation(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = BatchNormalization(name=bn_name)(x)
        x = Activation('relu')(x)
        return x

    return block


def Conv2D_Activation_BN(filters, kernel_size, padding='same', strides=(1, 1), name=None):
    def block(input_x):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = Conv2D(filters, kernel_size, padding=padding, strides=strides, name=conv_name)(input_x)
        x = Activation('relu')(x)
        x = BatchNormalization(name=bn_name)(x)
        return x

    return block


# Define Residual Block
def Residual_Block(filters, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    def block(input_x):
        x = Conv2D_BN_Activation(filters=filters, kernel_size=(3, 3), padding='same')(input_x)
        x = Conv2D_BN_Activation(filters=filters, kernel_size=(3, 3), padding='same')(x)
        # need convolution on shortcut for add different channel
        if with_conv_shortcut:
            shortcut = Conv2D_BN_Activation(filters=filters, strides=strides, kernel_size=kernel_size)(input_x)
            x = Add()([x, shortcut])
        else:
            x = Add()([x, input_x])
        return x

    return block


class MinimumEpochEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                 restore_best_weights=False, min_epoch=30):
        super(MinimumEpochEarlyStopping, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights)
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


class MinimumEpochReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, monitor='val_loss', min_delta=0., patience=0, verbose=0, mode='auto', factor=0.1, cooldown=0,
                 min_lr=0., min_epoch=30):
        super(MinimumEpochReduceLROnPlateau, self).__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr, )
        self.min_epoch = min_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.min_epoch:
            super().on_epoch_end(epoch, logs)


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
    plt.savefig(f"results/train_{version_num}.png")
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
    plt.savefig(f"results/val_{version_num}.png")
    plt.close(fig)


def get_dataset_dataframe(folder):
    dataset_path = pathlib.Path(folder)
    if not dataset_path.is_dir():
        raise Exception
    frames = []
    d: str
    for d in os.listdir(dataset_path):
        temp = pd.DataFrame({'file': (dataset_path / d).glob('*.png')})
        for i in range(6):
            temp[f'label_{i+1}'] = d[i]
            temp[f'label_{i+1}'] = temp[f'label_{i+1}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
        temp['label'] = d
        temp['file'] = temp['file'].astype('str')
        frames.append(temp)
    return pd.concat(frames, ignore_index=True)


def train(version_num, batch_size=64):
    training_dataset_dir = f"dataset"
    validate_dataset_dir = f"validate"
    checkpoint_path = f'checkpoints/{version_num}.hdf5'
    log_dir = f'logs/{version_num}'
    epochs = 100
    learning_rate = 0.0001
    optimizer = Adam(learning_rate)
    run = wandb.init(project="nthu_ccxp_captcha", entity="danielpclin", reinit=True, config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "version": version_num,
        "optimizer": optimizer._name
    })
    img_width = 80
    img_height = 30
    train_df = get_dataset_dataframe(training_dataset_dir)
    validate_df = get_dataset_dataframe(validate_dataset_dir)
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
    reduce_lr = MinimumEpochReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, cooldown=1, mode='auto',
                                              min_lr=0.00001, min_epoch=15)
    wandb_callback = WandbCallback()
    callbacks_list = [tensor_board, early_stop, checkpoint, reduce_lr, wandb_callback]
    # callbacks_list = [tensor_board, early_stop, reduce_lr, wandb_callback]

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
    with open(f"results/{version_num}.pickle", "wb") as file:
        pickle.dump(train_history.history, file)
    # with open(f"results/{version_num}.txt", "w") as file:
    #     loss_idx = np.nanargmin(train_history.history['val_loss'])
    #     file.write("Loss:\n")
    #     file.write(f"{train_history.history['val_loss'][loss_idx]}\n")
    #     acc = 1
    #     file.write("Accuracy:\n")
    #     for letter_idx in range(1, 13):
    #         acc *= train_history.history[f"val_label{letter_idx}_accuracy"][loss_idx]
    #     file.write(f"{acc}\n")

    plot(train_history.history, version_num)

    run.finish()
    K.clear_session()


def main():
    with open('current_version.txt', 'r+') as f:
        version_num = int(f.read())
        version_num += 1
        f.seek(0)
        f.write(str(version_num))
        f.truncate()
    train(version_num=version_num, batch_size=256)


if __name__ == "__main__":
    main()
