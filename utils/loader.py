import os
import cv2
import copy
import time
import torch
import random
import torchvision
import numpy as np
from PIL import Image
import torch.utils.data as data

import utils


class DataAugment:
    def __init__(self, debug=False):
        self.debug = debug

    def basic_matrix(self, translation):
        return np.array([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    def adjust_transform_for_image(self, img, trans_matrix):
        transform_matrix = copy.deepcopy(trans_matrix)
        height, width, channels = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply(self, img, trans_matrix):
        tmp_matrix = self.adjust_transform_for_image(img, trans_matrix)
        out_img = cv2.warpAffine(img, tmp_matrix[:2, :], dsize=(img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=0,)
        return out_img

    def random_vector(self, min, max):
        min = np.array(min)
        max = np.array(max)
        return np.random.uniform(min, max)

    def random_rotate(self, img, factor):
        angle = np.random.uniform(factor[0], factor[1])
        rotate_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        out_img = self.apply(img, rotate_matrix)
        return rotate_matrix, out_img

    def random_scale(self, img, min_translation, max_translation):
        factor = self.random_vector(min_translation, max_translation)
        scale_matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        out_img = self.apply(img, scale_matrix)
        return scale_matrix, out_img


class TrainDataPackage:
    def __init__(self, root="/home/shen/", transform=None, packaged=True):  # todo ./dataset
        self.training_file = "train.pt"
        self.aug = DataAugment(debug=True)
        self.packaged = packaged
        self.root = root
        self.num = 200
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(num_output_channels=1)])

        if not (os.path.exists(os.path.join(self.root, self.training_file))):
            print("No packaged dataset file (*.pt) in dataset/, Now generating...")
            self.generate()

        if packaged:
            self.train_data = torch.load(os.path.join(self.root, self.training_file))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        img = self.train_data[index]
        return img

    def generate(self):
        paths = [
            os.path.join(self.root, "BSD500/train/"),
            os.path.join(self.root, "BSD500/test/"),
            os.path.join(self.root, "BSD500/val/"),
        ]
        patches_list = []

        start = time.time()
        for path in paths:
            for roots, dirs, files in os.walk(path):
                if roots == path:
                    print("Image number: {}".format(files.__len__()))
                    for file in files:
                        if file[-4:] == ".jpg" or ".png":
                            temp = os.path.join(path, file)
                            print("=> Processing " + temp)
                            image = Image.open(temp)
                            patches = self.random_patch(image, self.num)
                            patches_list.extend(patches)
        print("Total patches: {}".format(patches_list.__len__()))
        print("Now Packaging...")
        with open(os.path.join(self.root, "train.pt"), 'wb') as f:
            torch.save(patches_list, f)
        end = time.time()
        print("Successfully packaged!, used time: {:.3f}".format(end - start))

    def random_patch(self, image, num):
        size = 96
        image = np.array(image, dtype=np.float32) / 255
        h, w = image.shape[0], image.shape[1]
        patches = []
        for n in range(num):
            max_h = random.randint(0, h - size)
            max_w = random.randint(0, w - size)
            patch = image[max_h:max_h + size, max_w:max_w + size]

            if 0 <= n <= 50:
                _, patch = self.aug.random_rotate(patch, (0., 1.))
            if 50 <= n <= 100:
                _, patch = self.aug.random_scale(patch, (1., 1.), (2., 2.))

            patch = self.transform(patch)
            patches.append(patch)
        return patches


def train_loader(batch_size):
    train_dataset = TrainDataPackage()
    dst = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                      shuffle=True, pin_memory=True, num_workers=8)
    return dst


if __name__ == '__main__':
    my_config = utils.GetConfig()
    TrainDataPackage()
    print("Now Loading train data...")
    dst = train_loader(my_config)
    print("Train data loaded, length: {}".format(dst.__len__()))
    for x in dst:
        print(x.size())
