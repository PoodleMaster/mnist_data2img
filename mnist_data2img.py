import os
import argparse
import shutil
from PIL import Image
from keras.datasets import mnist
from google.colab import drive
from tqdm import tqdm

drive.mount('/content/drive')

DIR_train = "/content/drive/My Drive/qiita_share/mnist_train_data"
DIR_test = "/content/drive/My Drive/qiita_share/mnist_test_data"
global args

#-------------------------------------------------------------------------------
# validation
#-------------------------------------------------------------------------------
def validation():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--new',
                        action="store_true",
                        help='new create option.')
    parser.add_argument('-d', '--debug',
                        action="store_true",
                        help='debug option.')
    args = parser.parse_args()


#-------------------------------------------------------------------------------
# argument
#-------------------------------------------------------------------------------
def argument():
    global args
    validation()
    if args.new:
        print("Create new png data from mnist.")
        deldir(DIR_train)
        deldir(DIR_test)


#-------------------------------------------------------------------------------
# debug print
#-------------------------------------------------------------------------------
def debug_print(data):
    global args
    if args.debug:
        print(type(data))
        print(data)

        
#-------------------------------------------------------------------------------
# del directory
#-------------------------------------------------------------------------------
def deldir(path):
    try:
        print("DEL TRY: ", path)
        shutil.rmtree(path)
        print("DEL COMP: ", path)
    except FileNotFoundError:
        print("DEL PASS: ", path)
        pass


#-------------------------------------------------------------------------------
# png save
#-------------------------------------------------------------------------------
def save(data, index, num, dirname):
    filename = dirname + "/" + str(num) + "/data" + "{0:05d}".format(index) + ".png"
    debug_print(filename)
    debug_print(data)
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[j, i] = int(data[i, j])
    img.save(filename)

# /content/drive/My Drive/qiita_share/mnist_train_data/5/data00000.png
# [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136 175  26 166 255 247 127   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253 225 172 253 242 195  64   0   0   0   0]
#  [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251  93  82  82  56  39   0   0   0   0   0]
#  [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119  25   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253 150  27   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252 253 187   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249 253 249  64   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253 253 207   2   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253 250 182   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201  78   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]


#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
def main():
    argument()

    # mnist data read #
    (train_data, train_label), (test_data, test_label) = mnist.load_data()

    # train #
    dirname = DIR_train
    if os.path.isdir(dirname) is False:
        os.mkdir(dirname)

    for i in range(10):
        dirname = DIR_train + "/" + str(i)
        if os.path.isdir(dirname) is False:
            os.mkdir(dirname)
    for i in tqdm(range(train_data.shape[0])):
        save(train_data[i], i, train_label[i], DIR_train)

    # test #
    dirname = DIR_test
    if os.path.isdir(dirname) is False:
        os.mkdir(dirname)

    for i in range(10):
        dirname = DIR_test + "/" + str(i)
        if os.path.isdir(dirname) is False:
            os.mkdir(dirname)
    for i in tqdm(range(test_data.shape[0])):
        save(test_data[i], i, test_label[i], DIR_test)

if __name__ == '__main__':
    main()
