import os
import sys
sys.path.append('../')
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

# from datasets.morphomnist import io
import io
from torchvision import transforms



# def load_data(
#     *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
# ):
#     """
#     For a dataset, create a generator over (images, kwargs) pairs.

#     Each images is an NCHW float tensor, and the kwargs dict contains zero or
#     more keys, each of which map to a batched Tensor of their own.
#     The kwargs dict can be used for class labels, in which case the key is "y"
#     and the values are integer tensors of class labels.

#     :param data_dir: a dataset directory.
#     :param batch_size: the batch size of each returned pair.
#     :param image_size: the size to which images are resized.
#     :param class_cond: if True, include a "y" key in returned dicts for class
#                        label. If classes are not available and this is true, an
#                        exception will be raised.
#     :param deterministic: if True, yield results in a deterministic order.
#     """

#     if not data_dir:
#         raise ValueError("unspecified data directory")
#     all_files = _list_image_files_recursively(data_dir)
#     classes = None
#     if class_cond:
#         # Assume classes are the first part of the filename,
#         # before an underscore.
#         class_names = [bf.basename(path).split("_")[0] for path in all_files]
#         sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
#         classes = [sorted_classes[x] for x in class_names]
#     dataset = ImageDataset(
#         image_size,
#         all_files,
#         classes=classes,
#         shard=MPI.COMM_WORLD.Get_rank(),
#         num_shards=MPI.COMM_WORLD.Get_size(),
#     )
#     if deterministic:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
#         )
#     else:
#         loader = DataLoader(
#             dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
#         )
#     while True:
#         yield from loader


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, split="train", deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if "celeba" in data_dir:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
        if deterministic:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
            )
        else:
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
            )
        while True:
            yield from loader
    
    if "morphomnist" in data_dir:  
        loader = get_dataloader_morphomnist(data_dir, batch_size, split_set=split, shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size())
    elif "pendulum" in data_dir:
        loader = get_dataloader_pendulum(data_dir, batch_size, split_set=split, shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size())
    elif "circuit" in data_dir:
        loader = get_dataloader_circuit(data_dir, batch_size, split_set=split, shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size())
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        return np.transpose(arr, [2, 0, 1]), out_dict





def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path


def load_morphomnist_like(root_dir, train: bool = True, columns=None) \
        -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        root_dir: path to data directory
        train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
        columns: list of morphometrics to load; by default (``None``) loads the image index and
            all available metrics: area, length, thickness, slant, width, and height
    Returns:
        images, labels, metrics
    """
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = io.load_idx(images_path)
    labels = io.load_idx(labels_path)

    if columns is not None and 'index' not in columns:
        usecols = ['index'] + list(columns)
    else:
        usecols = columns
    metrics = pd.read_csv(metrics_path, usecols=usecols, index_col='index')
    return images, labels, metrics


def save_morphomnist_like(images: np.ndarray, labels: np.ndarray, metrics: pd.DataFrame,
                          root_dir, train: bool):
    """
    Args:
        images: array of MNIST-like images
        labels: array of class labels
        metrics: data frame of morphometrics
        root_dir: path to the target data directory
        train: whether to save as the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
    """
    assert len(images) == len(labels)
    assert len(images) == len(metrics)
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    os.makedirs(root_dir, exist_ok=True)
    io.save_idx(images, images_path)
    io.save_idx(labels, labels_path)
    metrics.to_csv(metrics_path, index_label='index')


class MorphoMNISTLike(Dataset):
    def __init__(self, root_dir, train: bool = True, columns=None, shard=0, num_shards=1):
        """
        Args:
            root_dir: path to data directory
            train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
                subset (``False``, ``'t10k-*'`` files)
            columns: list of morphometrics to load; by default (``None``) loads the image index and
                all available metrics: area, length, thickness, slant, width, and height
        """
        self.root_dir = root_dir
        self.train = train
        images, labels, metrics_df = load_morphomnist_like(root_dir, train, columns)

        self.images = torch.as_tensor(images.copy())[shard:][::num_shards]
        self.labels = torch.as_tensor(labels.copy())[shard:][::num_shards]

        if columns is None:
            columns = metrics_df.columns

        # THE SHARDS THING HERE IS VERY IMPORTANT FOR CONDITIONING TO WORK!!!
        self.metrics = {col: torch.as_tensor(metrics_df[col])[shard:][::num_shards] for col in columns}
        self.columns = columns

        # normalization
        self.scale = {'thickness': [3.4, 2.4], 'intensity': [161, 94]}
        # Gaussian normalization
        self.gaussian_scale = {'thickness': [2.5, 0.63], 'intensity': [158.0, 48.4]}

        # assert len(self.images) == len(self.labels) and len(self.images) == len(metrics_df)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        scaled_item = {col: (values[idx] - self.scale[col][0]) / self.scale[col][1] for col, values in self.metrics.items()}
        
        # item = {col: (values[idx] - self.scale[col][0]) / self.scale[col][1] for col, values in self.metrics.items()}
        item = {col: values[idx] for col, values in self.metrics.items()}

        # item = {col: (values[idx] - self.gaussian_scale[col][0]) / self.gaussian_scale[col][1] for col, values in self.metrics.items()}

        # item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.images[idx].float() / 255.
        item['label'] = self.labels[idx]
        
        img = self.images[idx].float() / 255.
        img = img.unsqueeze(-1)
        # print(img.shape)
        # exit(0)
        out_dict = {}
        out_dict["y"] = np.array(self.labels[idx], dtype=np.int64)

        out_dict["c"] = np.array([item["thickness"], item["intensity"]], dtype=np.float32)

        return np.transpose(img, [2, 0, 1]), out_dict


def get_dataloader(dataset, config, split_set):
    if dataset == "morphomnist":
        loader = get_dataloader_morphomnist(config.data.path, config.sampling.batch_size, split_set=split_set)
    
    return loader


def get_dataloader_morphomnist(path, batch_size, split_set, shard, num_shards):
    assert split_set in ["train", "val", "test"]

    if split_set == "train":
        dataset = MorphoMNISTLike(root_dir=path,
                                columns=['thickness', 'intensity'], train=True, shard=shard, num_shards=num_shards)

        
        
    elif split_set == "val":
        dataset = MorphoMNISTLike(root_dir=path,
                                columns=['thickness', 'intensity'], train=False, shard=shard, num_shards=num_shards)
        
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset, 
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)], 
                                              generator=torch.Generator().manual_seed(42))

        dataset = split[1]
    else:
        dataset = MorphoMNISTLike(root_dir=path,
                                columns=['thickness', 'intensity'], train=False, shard=shard, num_shards=num_shards)
    
    # return torch.utils.data.DataLoader(dataset, 
    #                                    shuffle=False, 
    #                                    num_workers=1, 
    #                                    drop_last=True, 
    #                                    batch_size=batch_size, 
    #                                    sampler=DistributedSampler(dataset))
    # print(len(dataset))
    # exit(0)
    return torch.utils.data.DataLoader(dataset, 
                                       shuffle=True, 
                                       num_workers=1, 
                                       drop_last=True, 
                                       batch_size=batch_size)


class SyntheticLabeled(Dataset):
    def __init__(self, root, split="train", shard=0, num_shards=1):
        root = root + "/" + split

        imgs = os.listdir(root)

        self.dataset = split
        
        self.imgs = [os.path.join(root, k) for k in imgs][shard:][::num_shards]
        self.imglabel = [list(map(int,k[:-4].split("_")[1:]))  for k in imgs]
        
        self.imglabel = np.asarray(self.imglabel)[shard:][::num_shards]
        
        self.shard = shard
        self.num_shards = num_shards
        
        self.scale = np.array([[2,42],[104,44],[7.5, 4.5],[11,8]])
        # self.scale = np.array([[0,1],[0,1],[0,1],[0,1]])

        # self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        norm_label = torch.zeros(label.shape)        

        for i in range(4):
            norm_label[i] = (label[i] - self.scale[i][0]) / self.scale[i][1]

        pil_img = Image.open(img_path)
        data = np.asarray(pil_img)

        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,4)
            data = torch.from_numpy(pil_img)
        
        out_dict = {}
        out_dict["c"] = np.array(norm_label, dtype=np.float32)

        return data, out_dict
        
    def __len__(self):
        return len(self.imgs)
        

def get_dataloader_pendulum(path, batch_size, split_set, shard, num_shards):
    assert split_set in ["train", "val", "test"]

    if split_set == "train":
        dataset = SyntheticLabeled(path, split=split_set, shard=shard, num_shards=num_shards)       
    elif split_set == "test":
        dataset = SyntheticLabeled(path, split=split_set, shard=shard, num_shards=num_shards)

    # print(len(dataset))
    # exit(0)
    return torch.utils.data.DataLoader(dataset, 
                                       shuffle=True, 
                                       num_workers=1, 
                                       drop_last=True, 
                                       batch_size=batch_size)
    

class CausalCircuit(Dataset):
    def __init__(self, root, dataset="train", shard=0, num_shards=1):
        root = root + "/" + dataset
        
        self.imgs = []
        self.labels = []
        
        if dataset == "test":
            data = np.load(f'../datasets/causal_circuit/test.npz')
            self.img_labels = data['original_latents'][:, 0, :]
            
            
            # indices_11 = np.argwhere((self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4))
            # self.img_labels_1 = self.img_labels[(self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4)]
            # self.img_labels = self.img_labels_1
            
            temp = data['imgs'][:, 0]
            # filtered_images = np.take(temp, indices_11)
            
            # for i in range(len(filtered_images)):
            #     self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
            #     self.labels.append(self.img_labels[i])
            
            for i in range(len(temp)):
                self.imgs.append(Image.open(io.BytesIO(temp[i])))
                self.labels.append(self.img_labels[i])
            
        if dataset == "train":
            for k in range(5):
                data = np.load(f'../datasets/causal_circuit/train-{k}.npz')
                self.img_labels = data['original_latents'][:, 0, :]
                
                
                # indices_11 = np.argwhere((self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4))
                # self.img_labels_1 = self.img_labels[(self.img_labels[:, 0] > 0.4) | (self.img_labels[:, 1] > 0.4) | (self.img_labels[:, 2] > 0.4)]
                # self.img_labels = self.img_labels_1
                
                temp = data['imgs'][:, 0]
                # filtered_images = np.take(temp, indices_11)
                
                # for i in range(len(filtered_images)):
                #     self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                #     self.labels.append(self.img_labels[i])
                
                for i in range(len(temp)):
                    self.imgs.append(Image.open(io.BytesIO(temp[i])))
                    self.labels.append(self.img_labels[i])
                    
        self.labels = np.asarray(self.labels)[shard:][::num_shards]
        self.imgs = self.imgs[shard:][::num_shards]
        
        self.dataset = dataset
        self.transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        data = self.imgs[idx]
        # print(np.asarray(self.labels).reshape(35527, 4))
        perm = [3, 2, 1, 0]
        label = torch.from_numpy(np.asarray(self.labels)[idx][perm])


        if self.transforms:
            data = self.transforms(data)
        
        out_dict = {}
        out_dict["c"] = np.array(label, dtype=np.float32)
        
        return data, out_dict

    def __len__(self):
        return len(self.imgs)
    
    
def get_dataloader_circuit(path, batch_size, split_set, shard, num_shards):
    assert split_set in ["train", "val", "test"]
    if split_set == "train":
        dataset = CausalCircuit(path, split_set, shard=shard, num_shards=num_shards)      
    elif split_set == "test":
        dataset = CausalCircuit(path, split_set, shard=shard, num_shards=num_shards)



    return torch.utils.data.DataLoader(dataset, 
                                       shuffle=False, 
                                       num_workers=1, 
                                       drop_last=True, 
                                       batch_size=batch_size)
    
    
    
class CausalCircuitSimplified(Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset
        
        self.imgs = []
        self.labels = []
        
        if dataset == "train":
            for k in range(10):
                data = np.load(f'../../data/causal_data/causal_circuit/train-{k}.npz')

                perm = [3, 2, 1, 0]
                self.img_labels_0 = data['original_latents'][:, 0, :]
                self.img_labels_1 = data['original_latents'][:, 1, :]
                # self.img_labels = data['original_latents'][:, 0, :][:, perm]
                # THREE CASES

                self.img_labels = np.concatenate((self.img_labels_0, self.img_labels_1))
                # print(self.img_labels.shape)

                indices_11 = np.argwhere((self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2))
                self.img_labels_1 = self.img_labels[(self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2)]

                indices_12 = np.argwhere((self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2))
                self.img_labels_2 = self.img_labels[(self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2)]

                indices_13 = np.argwhere((self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2))
                self.img_labels_3 = self.img_labels[(self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2)]

                # indices_14 = np.argwhere((self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1))
                # self.img_labels_4 = self.img_labels[(self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1)]


                self.img_labels = np.concatenate((self.img_labels_1, self.img_labels_2, self.img_labels_3))
                # print(self.img_labels.shape)


                indices = np.concatenate((indices_11, indices_12, indices_13))

                # print(self.img_labels.shape)

                temp1 = data['imgs'][:, 0]
                temp2 = data['imgs'][:, 1]

                temp = np.concatenate((temp1, temp2))
                #filtered_images = temp
                filtered_images = np.take(temp, indices)
                # print(filtered_images)

                for i in range(len(filtered_images)):
                    self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                    self.labels.append(self.img_labels[i])

        else:
            data = np.load('../../data/causal_data/causal_circuit/test.npz')
            self.imgs = []
        
            perm = [3, 2, 1, 0]
            self.img_labels_0 = data['original_latents'][:, 0, :]
            self.img_labels_1 = data['original_latents'][:, 1, :]
            # self.img_labels = data['original_latents'][:, 0, :][:, perm]
            # THREE CASES

            self.img_labels = np.concatenate((self.img_labels_0, self.img_labels_1))
            print(self.img_labels.shape)

            indices_11 = np.argwhere((self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2))
            self.img_labels_1 = self.img_labels[(self.img_labels[:, 3] > 0.1) & (self.img_labels[:, 3] < 0.4) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 1] > 0.4) & (self.img_labels[:, 2] < 0.2)]

            indices_12 = np.argwhere((self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2))
            self.img_labels_2 = self.img_labels[(self.img_labels[:, 3] > 0.4) & (self.img_labels[:, 3] < 0.7) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] < 0.2) & (self.img_labels[:, 1] < 0.2)]

            indices_13 = np.argwhere((self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2))
            self.img_labels_3 = self.img_labels[(self.img_labels[:, 3] > 0.7) & (self.img_labels[:, 3] < 1) & (self.img_labels[:, 0] > 0.5) & (self.img_labels[:, 2] > 0.4) & (self.img_labels[:, 1] < 0.2)]

    #         indices_14 = np.argwhere((self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1))
    #         self.img_labels_4 = self.img_labels[(self.img_labels[:, 0] < 0.1) & (self.img_labels[:, 2] < 0.1) & (self.img_labels[:, 1] < 0.1)]


            self.labels = np.concatenate((self.img_labels_1, self.img_labels_2, self.img_labels_3))       
            indices = np.concatenate((indices_11, indices_12, indices_13))


            temp1 = data['imgs'][:, 0]
            temp2 = data['imgs'][:, 1]

            temp = np.concatenate((temp1, temp2))
            filtered_images = np.take(temp, indices)

            for i in range(len(filtered_images)):
                self.imgs.append(Image.open(io.BytesIO(filtered_images[i])))
                    
        self.dataset = dataset
        self.transforms = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        data = self.imgs[idx]
        # print(np.asarray(self.labels).reshape(35527, 4))
        perm = [3, 2, 1, 0]
        label = torch.from_numpy(np.asarray(self.labels)[idx][perm])

        if self.transforms:
            data = self.transforms(data)
        
        return data, label.float()

    def __len__(self):
        return len(self.imgs)