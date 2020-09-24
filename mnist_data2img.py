import os
import argparse
import shutil
from PIL import Image
from keras.datasets import mnist
from google.colab import drive

drive.mount('/content/drive')

DIR_train = "/content/drive/My Drive/qiita_share/mnist_train_data"
DIR_valid = "/content/drive/My Drive/qiita_share/mnist_valid_data"
global args

#-------------------------------------------------------------------------------
# validation
#-------------------------------------------------------------------------------
def validation():
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
    validation()
    if args.new:
        print("Create new png data from mnist.")
        deldir(DIR_valid)
        deldir(DIR_train)
    return args

        
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
def save(data, index, num, dir):
    debug_print(data)
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[j, i] = int(data[i,j])
    filename = dir + "/" + str(num) + "/test" + "{0:05d}".format(index) + ".png"
    img.save(filename)
    print(filename)

 
#-------------------------------------------------------------------------------
# debug print
#-------------------------------------------------------------------------------
def debug_print(data):
    if args.debug:
        print(type(data))
        print(data)
        print("-------------------------------------")

    
#-------------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------------
def main():
    argument()

    #### mnist data read ####
    (train_data, train_label), (valid_data, valid_label) = mnist.load_data()

    #### valid ####
    dirname = DIR_valid
    if os.path.isdir(dirname) is False:
        os.mkdir(dirname)

    for i in range(10):
        dirname = DIR_valid + "/" + str(i)
        if os.path.isdir(dirname) is False:
            os.mkdir(dirname)
    for i in range(valid_data.shape[0]):
        save(valid_data[i], i, valid_label[i], DIR_valid)

    #### train ####
    dirname = DIR_train
    if os.path.isdir(dirname) is False:
        os.mkdir(dirname)

    for i in range(10):
        dirname = DIR_train + "/" + str(i)
        if os.path.isdir(dirname) is False:
            os.mkdir(dirname)
    for i in range(train_data.shape[0]):
        save(train_data[i], i, train_label[i], DIR_train)


if __name__ == '__main__':
    main()
