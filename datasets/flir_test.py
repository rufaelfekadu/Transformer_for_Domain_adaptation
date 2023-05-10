from torchvision.io import read_image
import pandas as pd
import os
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ , idx= zip(*batch)
    # print('train collate fn' , imgs)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, idx

def val_collate_fn(batch):##### revised by luo
    imgs, pids, camids, viewids, img_paths, idx = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def source_target_train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    b_data = zip(*batch)
    # print('b_data is {}'.format(b_data))
    # if len(b_data) == 8:
    s_imgs, t_imgs, s_pids, t_pids, camids, viewids , s_file_name, t_file_name , s_idx, t_idx = b_data
    # print('make dataloader collate_fn {}'.format(pids))
    # print(pids)
    s_pid = torch.tensor(s_pids, dtype=torch.int64)
    t_pid = torch.tensor(t_pids, dtype=torch.int64)
    pids = (s_pid, t_pid)

    file_name = (s_file_name, t_file_name)
    
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    s_idx = torch.tensor(s_idx, dtype=torch.int64)
    t_idx = torch.tensor(t_idx, dtype=torch.int64)
    idx = (s_idx, t_idx)
    img1 = torch.stack(s_imgs, dim=0)
    img2 = torch.stack(t_imgs, dim=0)
    return (img1, img2), pids, camids, viewids, file_name, idx


class Coco(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, label_path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.label = pd.read_csv(label_path, delimiter=' ')
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.label.iloc[idx, 0])
        image = read_image(img_path)
        label = self.label.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def build_dataset(cfg):
    train_transforms = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    nb_classes=3
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_dataset = Coco(label_path='/nfs/users/ext_group7/project/CDTrans/data/train_labels.txt',
                root_dir='/nfs/users/ext_group7/project/CDTrans/data/cocoflir/', transform=train_transforms)
    
    train_set_normal = Coco(label_path='/nfs/users/ext_group7/project/CDTrans/data/train_labels.txt',
                root_dir='/nfs/users/ext_group7/project/CDTrans/data/cocoflir/', transform=val_transforms)
    
    val_dataset = Coco(label_path='/nfs/users/ext_group7/project/CDTrans/data/val_labels.txt',
                root_dir='/nfs/users/ext_group7/project/CDTrans/data/cocoflir/', transform=val_transforms)
            
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=train_collate_fn
    ) 

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )  
    return train_loader, train_loader_normal, val_loader, None, nb_classes, None, None