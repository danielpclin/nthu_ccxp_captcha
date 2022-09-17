A cnn model for captcha solving

Run `label_images.py` to label images and generate a list of codes

Run `generate_images.py` to generate dataset

Run `train.py` to train the model

Run `validate_images.py` to validate downloaded images


Run to convert into tensorflow.js format
```shell
docker run -it --rm -v "C:\Users\danielpclin\PycharmProjects\nthu_ccxp_captcha\tfjs:/tfjs" evenchange4/docker-tfjs-converter tensorflowjs_converter --input_format=keras tfjs/input/ccxp/1.hdf5 tfjs/output/ccxp/1
```

cuDNN 8.1
CUDA 11.2


Model
```
Conv 3x3, 64
Conv 3x3, 64
Conv 3x3, 64
MaxPool 2x2
Conv 3x3, 128
Conv 3x3, 128
Conv 3x3, 128
MaxPool 2x2
Conv 3x3, 256
Conv 3x3, 256
Conv 3x3, 256
MaxPool 2x2
Conv 3x3, 512
Conv 3x3, 512
Conv 3x3, 512
```
