import os
import pathlib

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend

alphabet = list('0123456789')
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


# Define callbacks
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
        else:
            logs["lr"] = backend.get_value(self.model.optimizer.lr)


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


def get_dataset_dataframe(folder, out):
    dataset_path = pathlib.Path(folder)
    if not dataset_path.is_dir():
        raise Exception
    frames = []
    d: str
    for d in os.listdir(dataset_path):
        frame = pd.DataFrame({'file': (dataset_path / d).glob('*.png')})
        for i in range(out):
            frame[f'label_{i+1}'] = d[i]
            frame[f'label_{i+1}'] = frame[f'label_{i+1}'].apply(lambda el: to_categorical(char_to_int[el], len(alphabet)))
        frame['label'] = d
        frame['file'] = frame['file'].astype('str')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
