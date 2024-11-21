import os
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):

    def get_class_names(self):
        return os.walk(self.data_set_path).__next__()[1]

    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        temp_class_name = class_names.copy()

        if not self.is_kfold:
            for index, class_name in enumerate(temp_class_name):
                label = index
                img_dir = os.path.join(self.data_set_path, class_name)
                img_files = os.walk(img_dir).__next__()[2]

                for img_file in img_files:
                    img_file = os.path.join(img_dir, img_file)
                    img = Image.open(img_file)
                    if img is not None:
                        all_img_files.append(img_file)
                        all_labels.append(label)

            return all_img_files, all_labels, len(all_img_files), len(class_names), temp_class_name

        for index, class_name in enumerate(temp_class_name):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            img_files_len = len(img_files)

            img_start_index = img_files_len // self.kfold_size

            if self.is_train:
                for idx in range(img_files_len):
                    if idx in range(img_start_index * self.kfold_index,
                                    min(img_start_index * (self.kfold_index + 1), img_files_len)):
                        continue
                    img_file = os.path.join(img_dir, img_files[idx])

                    img = Image.open(img_file)
                    # print("img :",type(img))
                    if img is not None:
                        all_img_files.append(img_file)
                        all_labels.append(label)
            else:
                for idx in range(img_start_index * self.kfold_index,
                                 min(img_start_index * (self.kfold_index + 1), img_files_len)):
                    img_file = os.path.join(img_dir, img_files[idx])

                    img = Image.open(img_file)

                    if img is not None:
                        all_img_files.append(img_file)
                        all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names), temp_class_name

    def __init__(self, data_set_path, kfold_index, kfold_size, is_kfold, is_train, transforms=None):
        self.kfold_index = kfold_index
        self.is_kfold = is_kfold
        self.kfold_size = kfold_size
        self.is_train = is_train
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes, self.class_names = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])

        image = image.convert("RGB")
        path = self.image_files_path[index]

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index], 'image_index': index, 'image_path': path}

    def __len__(self):
        return self.length


class Noise_Dataset(Dataset):

    def __init__(self, selected_path, transforms=None):
        self.selected_path = selected_path
        self.transforms = transforms
        self.length = len(selected_path)

    def __getitem__(self, index):
        if self.transforms is not None:
            image_noise = self.transforms(self.selected_path[index])
            print("image_type :", type(image_noise))

        return {'selected_image': image_noise}

    def __len__(self):
        return self.length
