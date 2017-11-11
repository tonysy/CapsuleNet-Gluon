import os 
import shutil
from data_dowloader import maybe_download, maybe_extract_tar

def download_and_creat_cifar10():
    url = 'http://www.cs.toronto.edu/~kriz/'
    filename = 'cifar-10-python.tar.gz'
    cifar10_dataset_location = './data/cifar10/'
    if not os.path.exists(cifar10_dataset_location):
        os.makedirs(cifar10_dataset_location)
    
    cifar10_gz = maybe_download(cifar10_dataset_location, url, filename)
    print('Cifar10 Dataset Download Complete')

    cifar10_untar = maybe_extract_tar(cifar10_dataset_location + filename, cifar10_dataset_location)
    print('Cifar10 Dataset Extraction Complete')


download_and_creat_cifar10()