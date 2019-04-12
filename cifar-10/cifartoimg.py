import os
import numpy as np
import imageio


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert(src, dstDir, num=1000):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)

    imgData = unpickle(src)

    for i in range(num):
        img = imgData['data'][i]
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)

        img_name = dstDir + str(i) + '.jpg'
        imageio.imwrite(img_name, img)


if __name__ == '__main__':
    root = '../data/cifar-10/sorted/'

    for i in range(10):
        src = root + 'train_' + str(i)
        dst = root + 'train_' + str(i) + '_img/'
        convert(src, dst)

    src = root + 'test'
    dst = root + 'testimg/'
    convert(src, dst)
