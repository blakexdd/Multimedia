import numpy as np
from skimage.io import imread, imsave, imshow
from skimage import img_as_float, img_as_ubyte
import tensorflow as tf
# Decreasing brights resolutiong function
# Arguments:
#   - Colored image and wanted bits for color
#     at resulted image
#       img : np.array, bits_for_color : int
# Returns:
#   - Grayscale reduced brights image
#       img
def decrease_brights_resolution(img: np.array, bits_for_color: int) -> np.array:
    # converting image colors to 0...1
    img = img_as_float(img)

    # converting image to grayscale
    img = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722

    # conveting image color back to 0..255
    img = img_as_ubyte(img)

    # finding amount of levels image will take
    # after brights convertion
    levels = 2 ** 8 / 2 ** bits_for_color

    # going through all image pixels and changing
    # to median value of range
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # getting range
            r = img[i, j] // levels

            # getting lower bound
            lower = r * levels - 1

            # getting upper bound
            upper = lower + levels

            # getting value of level
            value = lower + levels / 2

            # assigning value to image pixel
            img[i, j] = value

    return img


# Encoding to YUV format functuion
# Arguments:
#   - Image wanted to convert
#       img : np.array
# Returns:
#   - Image in YUV format
#       img
def encode_rgb(img: np.array) -> np.array:
    # converting colors to 0..1 range
    img_f = img_as_float(img)

    # making a copy of our image
    img_copy = img_f.copy()

    # converting rgb colors to yuv
    img_copy[:, :, 0] = img_f[:, :, 0] * 0.2126 + img_f[:, :, 1] * 0.7152 + img_f[:, :, 2] * 0.0722
    img_copy[:, :, 1] = - img_f[:, :, 0] * 0.0999 - img_f[:, :, 1] * 0.3360 + img_f[:, :, 2] * 0.4360
    img_copy[:, :, 2] = img_f[:, :, 0] * 0.6150 - img_f[:, :, 1] * 0.5586 - img_f[:, :, 2] * 0.0563

    # getting image width and height
    img_height = img_copy[:, :, 0].shape[0]
    img_weight = img_copy[:, :, 0].shape[1]

    # making pooling for u channel
    # reshaping image to 1, img_height, img_weight, 1 size
    # 1 stands for number of examples in batch, and 1 at last
    # position for number of channels
    img_input_u = tf.reshape(img_copy[:, :, 1], [1, img_height, img_weight, 1])

    # performing average pooling operation
    img_pooled_u = tf.nn.avg_pool(img_input_u, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    # reshaping image to initial size
    img_copy[:, :, 1] = tf.reshape(img_pooled_u, [img_height, img_weight])

    # making pooling for y channel
    # reshaping image to 1, img_height, img_weight, 1 size
    # 1 stands for number of examples in batch, and 1 at last
    # position for number of channels
    img_input_y = tf.reshape(img_copy[:, :, 2], [1, img_height, img_weight, 1])

    # performing average pooling operation
    img_pooled_y = tf.nn.avg_pool(img_input_y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    # reshaping image to initial size
    img_copy[:, :, 2] = tf.reshape(img_pooled_y, [img_height, img_weight])

    # conveting image colors to range 0..255
    img_copy = img_as_ubyte(img_copy)

    return img_copy


# Decoding from yuv to rgb function
# Arguments:
#   - Image wanted to convert
#       img : np.array
# Returns:
#   - Image in RGB format
#       img
def decode_yuv(img: np.array) -> np.array:
    # conveting image to 0..1
    img_copy = img_as_float(img)

    # making copy of image
    img_rgb = img_copy.copy()

    # converting image to rgb
    img_rgb[:, :, 0] = img_copy[:, :, 0] + 1.2803 * img_copy[:, :, 2]
    img_rgb[:, :, 1] = img_copy[:, :, 0] - 0.2148 * img_copy[:, :, 1] - 0.3805 * img_copy[:, :, 2]
    img_rgb[:, :, 2] = img_copy[:, :, 0] + 2.1279 * img_copy[:, :, 1]

    # clipping values from 0 to 1
    img_rgb = np.clip(img_rgb, 0, 1)

    # converting image back to 0..255
    img_rgb = img_as_ubyte(img_rgb)

    return img_rgb