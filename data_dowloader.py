from __future__ import print_function
import sys
import os
from six.moves.urllib.request import urlretrieve
import gzip
import shutil
import idx2numpy

last_percent_reported = None

'''
This File will download MNIST Dataset
and convert it into .mat format (28 x 28)
for using in the future
'''
url = 'http://yann.lecun.com/exdb/mnist/'
mnist_dataset_location = "./data/MNIST/"

def download_and_create_data() :
    url = 'http://yann.lecun.com/exdb/mnist/'
    mnist_dataset_location = "./data/MNIST/"
    if not os.path.exists(mnist_dataset_location):
        os.makedirs(mnist_dataset_location)

    train_images_zip = maybe_download(mnist_dataset_location, 'train-images-idx3-ubyte.gz')
    train_labels_zip = maybe_download(mnist_dataset_location, 'train-labels-idx1-ubyte.gz')

    test_images_zip = maybe_download(mnist_dataset_location, 't10k-images-idx3-ubyte.gz')
    test_labels_zip = maybe_download(mnist_dataset_location, 't10k-labels-idx1-ubyte.gz')

    print('MNIST Dataset Download Complete')

    train_images_file = maybe_extract(mnist_dataset_location + 'train-images-idx3-ubyte.gz')
    train_labels_file = maybe_extract(mnist_dataset_location + 'train-labels-idx1-ubyte.gz')

    test_images_file = maybe_extract(mnist_dataset_location + 't10k-images-idx3-ubyte.gz')
    test_labels_file = maybe_extract(mnist_dataset_location + 't10k-labels-idx1-ubyte.gz')

    print('MNIST Dataset Extraction Complete')

    train_images = idx2numpy.convert_from_file(mnist_dataset_location + 'train-images-idx3-ubyte')
    train_label = idx2numpy.convert_from_file(mnist_dataset_location + 'train-labels-idx1-ubyte')

    test_images = idx2numpy.convert_from_file(mnist_dataset_location + 't10k-images-idx3-ubyte')
    test_label = idx2numpy.convert_from_file(mnist_dataset_location + 't10k-labels-idx1-ubyte')

    return train_images, train_label, test_images, test_label

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intend for users with slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        
        last_percent_reported = percent
    
def maybe_download(path, filename, force=False):
    """Download a file if not present, and make sure it's the right size"""
    if force or not os.path.exists(path + filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url+filename, path+filename, reporthook=download_progress_hook)
        print('\nDownload Completes!')
    else:
        filename = path + filename
    statinfo = os.stat(filename)
    return filename

def maybe_extract(filename, force=False):
    outfile = filename[:-3]
    if os.path.exists(outfile) and not force:
        # You may override by setting force = True.
        print('%s already present - Skipping extraction of %s.' % (outfile, filename))

    else:
        print('Extracting data for %s.' % outfile)
        inF = gzip.open(filename, 'rb')
        outF = open(outfile, 'wb')
        outF.write(inF.read())
        inF.close()
    
    data_folders = outfile
    print(data_folders)

    return data_folders