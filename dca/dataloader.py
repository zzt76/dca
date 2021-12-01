import torch
import torch.utils.data
import torchvision.transforms.functional as tf
import random
from PIL import Image


class DataLoader():
    def __init__(self, dataset, height, width, batch_size, num_workers, train_list, val_list, test_list, run_train, do_aug):
        ### Initialize ###
        self.dataset = dataset
        self.height = height
        self.width = width
        self.batchsize = batch_size
        self.num_workers = num_workers
        self.run_train = run_train
        self.do_aug = do_aug
        ### datalist ###
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list

    def get_train_val_loaders(self):
        train_loader = val_loader = None
        print(f"Loading training data from {self.train_list}")
        train_loader = self.generate_loader(self.train_list)
        print(f"Loading validation data from {self.val_list}")
        val_loader = self.generate_loader(self.val_list, is_val=True)
        return train_loader, val_loader

    def get_test_loader(self):
        print(f"Loading testing data from {self.test_list}")
        test_loader = self.generate_loader(self.test_list)
        return test_loader

    def generate_loader(self, file_list, is_val=False):
        dataset = DatasetGenerator(self.dataset, file_list, self.width, self.height, self.run_train, is_val, self.do_aug)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batchsize,
            shuffle=self.run_train,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader


class DatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, dataset, file_list, width=640, height=480, run_train=False, is_val=False, do_aug=True):
        super(DatasetGenerator, self).__init__()
        self.dataset = dataset
        self.run_train = run_train
        self.is_val = is_val
        self.do_aug = do_aug
        with open(file_list, 'r') as f:
            self.list = f.readlines()
        self.size = len(self.list)
        self.h = height
        self.w = width

    def __getitem__(self, index):
        item = self.list[index]
        item = item.replace('\n', '')
        if self.dataset == 'vari':
            length = len(item.split(','))
            assert length == 4, f'Number of item{item} must be 4.'
            color_path, _, _, depth_path = item.split(',')
            name = color_path.split('/')[-1].split('.')[0]

            color = tf.to_tensor(Image.open(color_path))
            depth = tf.to_tensor(tf.to_grayscale(Image.open(depth_path)))
            depth = tf.invert(depth) * 10.0

            if self.run_train:
                color, depth = self.random_rotate([color, depth])
                color, depth = self.random_resize_crop([color, depth], self.h, self.w)
                if self.do_aug:
                    color, depth = self.augmentation([color, depth], p_f=0.5, p_a=0.9)

            return {'color': color, 'depth': depth, 'name': name}

        elif self.dataset == 'nyu':
            paths = item.split(',')
            color_path, depth_path = paths[:2]
            # get name from path
            name = color_path.split('/')[-2] + '-' + color_path.split('/')[-1].split('.')[0].split('_')[-1]

            if self.run_train and not self.is_val:
                # To avoid blank boundaries due to pixel registration, after crop the image size is [420, 560]
                dense_path = paths[2]
                color = tf.to_tensor(Image.open(color_path).crop((45, 41, 605, 461)))
                depth = tf.to_tensor(Image.open(depth_path).crop((45, 41, 605, 461)))  # crop to [420, 560]
                dense = tf.to_tensor(Image.open(dense_path).crop((45, 41, 605, 461)))
                depth = depth.div(1000)  # orignial range 0 ~ 10000, after div depth range [0,10]
                dense = dense.div(1000)
                dense = dense * (depth.max() / dense.max())

                color, depth, dense = self.random_rotate([color, depth, dense])
                color, depth, dense = self.random_resize_crop([color, depth, dense], self.h, self.w)  # crop to [416, 544]

                if self.do_aug:
                    color, depth, dense = self.augmentation([color, depth, dense], p_f=0.5, p_a=0.9)

                return {'color': color, 'depth': depth, 'dense': dense, 'name': name}

            else:
                color = tf.to_tensor(Image.open(color_path))
                depth = tf.to_tensor(Image.open(depth_path))
                depth = depth.div(1000)
                if self.is_val:
                    color = tf.center_crop(color, [416, 544])
                    depth = tf.center_crop(depth, [416, 544])

                return {'color': color, 'depth': depth, 'name': name}

    def random_rotate(self, images: list, min=-3, max=3):
        rotate_angle = random.random()*(max-min) - (max-min)/2
        for i, image in enumerate(images):
            images[i] = tf.rotate(image, rotate_angle, tf.InterpolationMode.BILINEAR)
        return images

    def augmentation(self, images: list, p_f=0.5, p_a=0.9):
        '''Please put color image in image[0]'''
        # Random crop(not implemented)
        # use tf.resized_crop
        # Random flipping
        do_flip = random.random()
        for i, image in enumerate(images):
            if do_flip < p_f:
                images[i] = tf.hflip(image)

        # Random gamma, brightness, color augmentation, only for color image, image[0]!!
        do_aug = random.random()
        if do_aug < p_a:
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            images[0] = tf.adjust_gamma(images[0], gamma)
            # brightness augmentation
            if self.dataset == 'vari':
                # Which will cause weird results in vari dataset
                brightness = random.uniform(0.8, 1.2)
            elif 'nyu' in self.dataset:
                brightness = random.uniform(0.7, 1.25)
            images[0] = tf.adjust_brightness(images[0], brightness)
            # contrast augmentation
            images[0] = tf.adjust_contrast(images[0], random.uniform(0.9, 1.4))
            # sharpness augmentation
            images[0] = tf.adjust_sharpness(images[0], random.uniform(0.9, 1.4))

        return images

    def random_resize_crop(self, images: list, height, width):
        y = random.randint(0, images[0].shape[1] - height)
        x = random.randint(0, images[0].shape[2] - width)
        for i, image in enumerate(images):
            assert image.shape[1] >= height
            assert image.shape[2] >= width
            images[i] = image[:, y:y + height, x:x + width]
        return images

    def __len__(self):
        return self.size
