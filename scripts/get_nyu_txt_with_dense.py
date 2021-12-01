import os
from os.path import isfile
import random

train_path = '../nyu_dataset/sync'
test_path = '../nyu_dataset/official_splits/test'
txt_path = './data/nyu_raw'


def get_original_txt(path):
    txt = open(txt_path+'/' + 'original.txt', mode='w')
    folders = os.listdir(path)
    for folder in folders:
        numbers = []
        names = os.listdir(os.path.join(path, folder))
        for name in names:
            if 'dense' not in name:
                number = int(name.split('.')[0].split('_')[-1])
                numbers.append(number)
        numbers.sort()
        numbers = list(set(numbers))
        for num in numbers:
            num = '%05d' % num
            rgb_path = path+'/'+folder+'/'+'rgb_'+str(num)+'.jpg'
            raw_depth_path = path+'/'+folder+'/'+'sync_depth_'+str(num)+'.png'
            dense_depth_path = path+'/'+folder+'/'+'dense'+'/'+'sync_depth_dense'+'_' + str(num)+'.png'
            assert isfile(rgb_path) and isfile(raw_depth_path) and isfile(dense_depth_path), FileNotFoundError
            txt.write(rgb_path + ',' + raw_depth_path + ',' + dense_depth_path + '\n')
    print(f'Generated original.txt in {txt_path}')
    txt.close()


def get_test_txt(test_path):
    txt = open(txt_path+'/' + 'test.txt', mode='w')
    scenes = os.listdir(test_path)
    paths = []
    for scene in scenes:
        files = os.listdir(os.path.join(test_path, scene))
        for file in files:
            if 'dense' not in file and 'sync_depth' not in file:
                num = file.split('.')[0].split('_')[-1]
                rgb_path = test_path + '/' + scene + '/' + 'rgb_' + num + '.jpg'
                depth_path = test_path + '/' + scene + '/' + 'sync_depth_' + num + '.png'
                assert isfile(rgb_path) and isfile(depth_path), FileExistsError
                path = rgb_path + ',' + depth_path + '\n'
                if path not in paths:
                    paths.append(path)
    txt.writelines(paths)
    print(f'Generated test.txt in {txt_path}')


def split_train_val(path, num_sample=200):
    with open(path, mode='r') as origin:
        original_list = origin.readlines()
        random.shuffle(original_list)
        train_list = original_list[:-num_sample]
        train = open(txt_path+'/' + 'train.txt', mode='w')
        train.writelines(train_list)
        print(f'Generated train.txt in {txt_path}')
        val_list = original_list[-num_sample:]
        val = open(txt_path+'/' + 'val.txt', mode='w')
        val.writelines(val_list)
        print(f'Generated val.txt in {txt_path}')


if __name__ == '__main__':
    get_original_txt(train_path)
    get_test_txt(test_path)
    split_train_val(os.path.join(txt_path, 'original.txt'), num_sample=200)
