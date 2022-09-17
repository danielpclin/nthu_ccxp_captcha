import os

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model, models

from helpers import get_dataset_dataframe, int_to_char

# os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)


def predict_ccxp_oauth(batch_size=500):
    predict_dir = "dataset/oauth/validate"
    predict_df = get_dataset_dataframe(predict_dir, out=4)
    model_file = f'checkpoints/oauth/4.hdf5'
    img_width, img_height = 150, 80
    # img_width, img_height = 75, 40
    datagen = ImageDataGenerator(rescale=1. / 255)
    # predict_generator = datagen.flow_from_dataframe(dataframe=predict_df, directory='.', x_col="file",
    #                                                 y_col=[f'label_{i}' for i in range(1, 5)],
    #                                                 class_mode="multi_output",
    #                                                 target_size=(img_height, img_width),
    #                                                 batch_size=batch_size)
    predict_generator = datagen.flow_from_dataframe(dataframe=predict_df, directory='.', x_col="file",
                                                    class_mode=None, shuffle=False, validate_filenames=False,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size)
    model = models.load_model(model_file)
    pred = model.predict(
        predict_generator,
        steps=predict_generator.n // predict_generator.batch_size,
        verbose=1,
    )
    result = ["" for _ in range(len(pred[0]))]
    pred = np.argmax(pred, axis=2)
    for digit in pred:
        for index, code in enumerate(digit):
            result[index] = result[index] + int_to_char[code]
    predict_df['pred'] = result
    predict_df.to_csv(f'predict_oauth.csv', index=False, columns=['file', 'label', 'pred'])
    predict_df.loc[predict_df['label'] != predict_df['pred']].to_csv(f'predict_oauth_diff.csv', index=False, columns=['file', 'label', 'pred'])


if __name__ == "__main__":
    predict_ccxp_oauth()
