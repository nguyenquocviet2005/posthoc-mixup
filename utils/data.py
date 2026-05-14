import os
import numpy as np
import torch
import sys
sys.path.append('../')
import torchvision as tv
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import csv, torchvision, random, os
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict


CHEXPERT_LABEL_COLUMNS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            len_train = int(np.floor(0.9 * len(self.dataset)))
            return (len_train+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    def __init__(self, dataset, dataname, indices=None):
        self.dataname = dataname
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices
        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


def get_loader(data, data_path, batch_size, args):
    # dataset normalize values
    if data == 'pathmnist':
        try:
            from medmnist import PathMNIST
        except ImportError as exc:
            raise ImportError("PathMNIST requires medmnist. Install it with: pip install medmnist") from exc
        os.makedirs(data_path, exist_ok=True)

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        train_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        test_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        train_base = PathMNIST(
            split='train',
            transform=train_transforms,
            download=True,
            as_rgb=True,
            root=data_path,
            size=224,
        )
        eval_base = PathMNIST(
            split='val',
            transform=test_transforms,
            download=True,
            as_rgb=True,
            root=data_path,
            size=224,
        )
        test_base = PathMNIST(
            split='test',
            transform=test_transforms,
            download=True,
            as_rgb=True,
            root=data_path,
            size=224,
        )

        train_data = MedMNISTWithIdx(train_base)
        eval_data = MedMNISTWithIdx(eval_base)
        test_data = MedMNISTWithIdx(test_base)

        test_label = test_data.targets
        test_onehot = np.eye(9)[test_label]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        sample, _, _ = train_data[0]
        print("-------------------Make loader-------------------")
        print(f"PathMNIST MedMNIST+ size=224 | as_rgb=True | sample tensor shape={tuple(sample.shape)}")
        print('Train Dataset :', len(train_loader.dataset), 'Valid Dataset :', len(valid_loader.dataset),
              '   Test Dataset :', len(test_loader.dataset))
        return train_loader, valid_loader, test_loader, test_onehot, test_label

    if data == 'chexpert_small':
        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        train_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        test_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        target = getattr(args, 'chexpert_target', 'Pleural Effusion')
        uncertain_policy = getattr(args, 'chexpert_uncertain', 'one')
        frontal_only = getattr(args, 'chexpert_frontal_only', False)

        train_data = CheXpertSmallDataset(
            data_path,
            'train.csv',
            target,
            transform=train_transforms,
            uncertain_policy=uncertain_policy,
            frontal_only=frontal_only,
        )
        eval_data = CheXpertSmallDataset(
            data_path,
            'valid.csv',
            target,
            transform=test_transforms,
            uncertain_policy=uncertain_policy,
            frontal_only=frontal_only,
        )
        test_data = CheXpertSmallDataset(
            data_path,
            'valid.csv',
            target,
            transform=test_transforms,
            uncertain_policy=uncertain_policy,
            frontal_only=frontal_only,
        )

        test_onehot = np.eye(2)[test_data.targets]
        test_label = test_data.targets

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

        print("-------------------Make loader-------------------")
        print(f"CheXpert target: {target} | uncertain: {uncertain_policy} | frontal_only: {frontal_only}")
        print('Train Dataset :', len(train_loader.dataset), 'Valid Dataset :', len(valid_loader.dataset),
              '   Test Dataset :', len(test_loader.dataset))
        return train_loader, valid_loader, test_loader, test_onehot, test_label

    if data == 'chest_xray':
        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]
        
        train_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        test_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        class ImageFolderWithIdx(Dataset):
            def __init__(self, root, transform=None):
                self.dataset = datasets.ImageFolder(root=root, transform=transform)
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                img, target = self.dataset[idx]
                return img, target, idx

        train_data = ImageFolderWithIdx(root=os.path.join(data_path, 'train'), transform=train_transforms)
        eval_data = ImageFolderWithIdx(root=os.path.join(data_path, 'val'), transform=test_transforms)
        test_data = ImageFolderWithIdx(root=os.path.join(data_path, 'test'), transform=test_transforms)
        
        test_targets = test_data.dataset.targets
        test_onehot = one_hot_encoding(test_targets)
        test_label = test_targets
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print("-------------------Make loader-------------------")
        print('Train Dataset :', len(train_loader.dataset), 'Valid Dataset :', len(valid_loader.dataset),
              '   Test Dataset :', len(test_loader.dataset))
        return train_loader, valid_loader, test_loader, test_onehot, test_label

    if data == 'mri_tumor':
        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]
        
        train_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        test_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        train_dir = os.path.join(data_path, 'Training')
        test_dir = os.path.join(data_path, 'Testing')
        
        full_train_set = datasets.ImageFolder(root=train_dir)
        num_total = len(full_train_set)
        indices = list(range(num_total))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_val = int(np.floor(0.1 * num_total))
        val_idx = indices[:split_val]
        train_idx = indices[split_val:]
        
        class ImageFolderSubset(Dataset):
            def __init__(self, full_set, indices, transform=None):
                self.full_set = full_set
                self.indices = indices
                self.transform = transform
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                path, target = self.full_set.samples[real_idx]
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, target, idx
                
        train_data = ImageFolderSubset(full_train_set, train_idx, train_transforms)
        eval_data = ImageFolderSubset(full_train_set, val_idx, test_transforms)
        
        class ImageFolderWithIdx(Dataset):
            def __init__(self, root, transform=None):
                self.dataset = datasets.ImageFolder(root=root, transform=transform)
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                img, target = self.dataset[idx]
                return img, target, idx
                
        test_data = ImageFolderWithIdx(root=test_dir, transform=test_transforms)
        
        test_targets = test_data.dataset.targets
        test_onehot = one_hot_encoding(test_targets)
        test_label = test_targets
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print("-------------------Make loader-------------------")
        print('Train Dataset :', len(train_loader.dataset), 'Valid Dataset :', len(valid_loader.dataset),
              '   Test Dataset :', len(test_loader.dataset))
        return train_loader, valid_loader, test_loader, test_onehot, test_label

    if data == 'skin_cancer_isic':
        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]
        
        train_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        test_transforms = tv.transforms.Compose([
            tv.transforms.Resize((224, 224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        full_set = datasets.ImageFolder(root=data_path)
        
        num_total = len(full_set)
        indices = list(range(num_total))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_val = int(np.floor(0.1 * num_total))
        split_test = int(np.floor(0.1 * num_total))
        
        test_idx = indices[:split_test]
        val_idx = indices[split_test:split_test+split_val]
        train_idx = indices[split_test+split_val:]
        
        class ImageFolderSubset(Dataset):
            def __init__(self, full_set, indices, transform=None):
                self.full_set = full_set
                self.indices = indices
                self.transform = transform
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                path, target = self.full_set.samples[real_idx]
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, target, idx
                
        train_data = ImageFolderSubset(full_set, train_idx, train_transforms)
        eval_data = ImageFolderSubset(full_set, val_idx, test_transforms)
        test_data = ImageFolderSubset(full_set, test_idx, test_transforms)
        
        test_targets = [full_set.samples[i][1] for i in test_idx]
        test_onehot = one_hot_encoding(test_targets)
        test_label = test_targets
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print("-------------------Make loader-------------------")
        print('Train Dataset :', len(train_loader.dataset), 'Valid Dataset :', len(valid_loader.dataset),
              '   Test Dataset :', len(test_loader.dataset))
        return train_loader, valid_loader, test_loader, test_onehot, test_label

    if data == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
    elif data == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]

    # augmentation
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # load datasets
    if data == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                      train=True,transform=train_transforms,download=True)
        eval_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False,transform=test_transforms,download=False)
        test_set = datasets.CIFAR100(root=os.path.join(data_path, 'cifar100_data'),
                                     train=False, transform=test_transforms, download=False)

    elif data == 'cifar10':
        train_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                     train=True,transform=train_transforms,download=True)
        eval_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False,transform=test_transforms,download=False)
        test_set = datasets.CIFAR10(root=os.path.join(data_path, 'cifar10_data'),
                                    train=False, transform=test_transforms, download=False)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    X_train_total = np.array(train_set.data)
    Y_train_total = np.array(train_set.targets)
    X_train = X_train_total[train_idx]
    X_valid = X_train_total[val_idx]
    Y_train = Y_train_total[train_idx]
    Y_valid = Y_train_total[val_idx]
    train_set.data = X_train.astype('uint8')
    train_set.targets = Y_train
    eval_set.data = X_valid.astype('uint8')
    eval_set.targets = Y_valid

    method = None
    train_data = Custom_Dataset(train_set.data,train_set.targets,'cifar', train_transforms, method=method)
    eval_data = Custom_Dataset(eval_set.data, eval_set.targets,'cifar', test_transforms)
    test_data = Custom_Dataset(test_set.data, test_set.targets, 'cifar', test_transforms)
    test_onehot = one_hot_encoding(test_set.targets)
    test_label = test_set.targets

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=4)
    valid_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=4)

    print("-------------------Make loader-------------------")
    print('Train Dataset :',len(train_loader.dataset), 'Valid Dataset :',len(valid_loader.dataset),
          '   Test Dataset :',len(test_loader.dataset))
    return train_loader, valid_loader, test_loader, test_onehot, test_label


class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None, method=None):
        self.x_data = x
        self.y_data = y
        self.data = data_set
        self.transform = transform
        self.method = method

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        if self.method == None:
            x = self.transform(img)
        else:
            x = img
        return x, self.y_data[idx], idx


class MedMNISTWithIdx(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        if hasattr(dataset, 'labels'):
            self.targets = [self._to_int_label(label) for label in dataset.labels]
        else:
            self.targets = [self._to_int_label(dataset[i][1]) for i in range(len(dataset))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return img, self._to_int_label(target), idx

    @staticmethod
    def _to_int_label(target):
        if isinstance(target, np.ndarray):
            return int(target.reshape(-1)[0])
        if torch.is_tensor(target):
            return int(target.reshape(-1)[0].item())
        if isinstance(target, (list, tuple)):
            return int(target[0])
        return int(target)


class CheXpertSmallDataset(Dataset):
    def __init__(self, root, csv_name, target, transform=None, uncertain_policy='one', frontal_only=False):
        if target not in CHEXPERT_LABEL_COLUMNS:
            raise ValueError(f"Unknown CheXpert target '{target}'. Choose one of: {CHEXPERT_LABEL_COLUMNS}")
        if uncertain_policy not in ('zero', 'one', 'ignore'):
            raise ValueError("chexpert_uncertain must be one of: zero, one, ignore")

        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []
        csv_path = self._resolve_csv_path(root, csv_name)

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if frontal_only and row.get('Frontal/Lateral') != 'Frontal':
                    continue

                label = self._parse_label(row.get(target, ''), uncertain_policy)
                if label is None:
                    continue

                image_path = self._resolve_image_path(root, row['Path'])
                self.samples.append((image_path, label))
                self.targets.append(label)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No CheXpert samples loaded from {csv_path}. Check data_path, target, and filters."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, idx

    @staticmethod
    def _resolve_csv_path(root, csv_name):
        candidates = [
            os.path.join(root, csv_name),
            os.path.join(root, 'CheXpert-v1.0-small', csv_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find {csv_name} under {root}")

    @staticmethod
    def _resolve_image_path(root, csv_image_path):
        rel_path = csv_image_path
        prefix = 'CheXpert-v1.0-small/'
        candidates = [
            os.path.join(root, rel_path),
            os.path.join(root, rel_path.replace(prefix, '', 1)) if rel_path.startswith(prefix) else None,
            os.path.join(root, 'CheXpert-v1.0-small', rel_path.replace(prefix, '', 1))
            if rel_path.startswith(prefix) else None,
        ]
        for path in candidates:
            if path is not None and os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find CheXpert image '{csv_image_path}' under {root}")

    @staticmethod
    def _parse_label(value, uncertain_policy):
        if value == '' or value is None:
            return 0
        label = float(value)
        if label == -1.0:
            if uncertain_policy == 'ignore':
                return None
            return 1 if uncertain_policy == 'one' else 0
        return 1 if label == 1.0 else 0


def one_hot_encoding(label):
    print("one_hot_encoding process")
    cls = sorted(set(label))
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot

