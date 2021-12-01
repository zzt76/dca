import os
from os.path import isfile
import random
import tqdm

train_path = '../train_dataset/'
test_path = '../test_dataset/'
txt_path = './data/vari'

lights = ["Environment", "Indoor",
          "SunNoon_Indoor", "SunNight_Indoor", "SunMorning_Indoor",
          "SunNoon_Indoor_Environment", "SunNight_Indoor_Environment", "SunMorning_Indoor_Environment",
          "SunNoon", "SunNight", "SunMorning"]
geometry_types = ["CGeometry_NormalsGeometry", "CShading_Albedo", "CGeometry_ZDepth"]
# geometry_types = ["CGeometry_ZDepth"]


def get_txt(path, txt_name):
    names = get_image_names(path)
    lines = []
    for name in tqdm.tqdm(names):
        folder, cam, num = name
        img_dir = os.path.join(path, folder, cam)
        for light in lights:
            color_name = folder + '_' + cam + '_' + light + num + '.jpg'
            color_path = os.path.join(img_dir, color_name)
            img_paths = [color_path]
            for geo in geometry_types:
                if geo == 'CGeometry_ZDepth':
                    geo_name = folder + '_' + cam + '_' + geo + num + '.tif'
                else:
                    geo_name = folder + '_' + cam + '_' + geo + num + '.jpg'
                geo_path = os.path.join(img_dir, geo_name)
                img_paths.append(geo_path)
            for img in img_paths:
                assert os.path.exists(img), f'File {img} not exists!'
            img_line = ','.join(i for i in img_paths) + '\n'
            lines.append(img_line)
    txt = open(txt_path+'/' + txt_name, mode='w')
    txt.writelines(lines)
    txt.close()


def get_image_names(path):
    folders = os.listdir(path)
    img_names = []
    for folder in folders:
        cam_dir = os.path.join(path, folder)
        cameras = os.listdir(cam_dir)
        for cam in cameras:
            image_dir = os.path.join(cam_dir, cam)
            images = os.listdir(image_dir)
            numbers = []
            for image in images:
                number = image.split('.')[0][-4:]
                number = int(number)
                if number not in numbers:
                    numbers.append(number)
            numbers.sort()
            for num in numbers:
                num = "%04d" % num
                img_names.append((folder, cam, num))
    return img_names


def get_val_from_train(path, num_sample=200):
    path = os.path.join(path, 'original.txt')
    with open(path, mode='r') as train:
        original_list = train.readlines()
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
    get_txt(train_path, 'original.txt')
    get_txt(test_path, 'test.txt')
    get_val_from_train(txt_path, num_sample=200)
