from typing import Any, Tuple
import torch
from utils.meter import AverageMeter
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import Discriminator, make_model
from config import cfg

class CustomImageDataset(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index), self.imgs[index][0]

def train_epoch(model, source_loader, taget_loader, optimizer, criterion, device, epoch, print_freq=1000):
    model.train()
    losses = AverageMeter()
    iters = min(len(source_loader), len(taget_loader))
    source_iter, target_iter = iter(source_loader), iter(taget_loader)

    for i in range(iters):

        source_data, source_label, _ = source_iter.next()
        target_data, target_label, _ = target_iter.next()

        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data, target_label = target_data.to(device), target_label.to(device)

        source_domain_label = torch.zeros(source_data.shape[0]).long().to(device)
        target_domain_label = torch.ones(target_data.shape[0]).long().to(device)

        source_domain_pred = model(source_data)
        target_domain_pred = model(target_data)

        total_pred = torch.cat([source_domain_pred, target_domain_pred], dim=0)
        total_label = torch.cat([source_domain_label, target_domain_label], dim=0)

        loss = criterion(total_pred, total_label)
        losses.update(loss.item(), source_data.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, i, iters, loss=losses.avg))
            
def train(model, source_loader, taget_loader, optimizer, criterion, device, epochs, print_freq=1000):
    for epoch in range(epochs):
        train_epoch(model, source_loader, taget_loader, optimizer, criterion, device, epoch, print_freq=print_freq)
        loss, pred = validate(model, source_loader, taget_loader, criterion, device)
        print(f'Epoch: [{epoch}]\t Validation Loss {loss:.4f}\t validation Accuracy {pred:.4f}\t')

def validate(model, source_loader, taget_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accuracy = AverageMeter()
    iters = min(len(source_loader), len(taget_loader))
    source_iter, target_iter = iter(source_loader), iter(taget_loader)

    with torch.no_grad():
        for i in range(iters):

            source_data, source_label, source_img_path = source_iter.next()
            target_data, target_label, target_img_path = target_iter.next()

            source_data = source_data.to(device)
            target_data = target_data.to(device)
            total_data = torch.cat([source_data, target_data], dim=0)

            source_domain_label = torch.zeros(source_data.shape[0]).long().to(device)
            target_domain_label = torch.ones(target_data.shape[0]).long().to(device)
            total_label = torch.cat([source_domain_label, target_domain_label], dim=0)

            # source_domain_pred = model(source_data)
            # target_domain_pred = model(target_data)
            total_pred_conf = model(total_data)
            total_pred =  torch.argmax(total_pred_conf, dim=1)
            accuracy.update(torch.sum(total_pred == total_label).item(), total_label.shape[0])

            # total_pred_conf = torch.cat([source_domain_pred, target_domain_pred], dim=0)
            

            loss = criterion(total_pred_conf, total_label)
            losses.update(loss.item(), source_data.size(0))

    return losses.avg, accuracy.avg

def infer_target(model, target_loader, device, split='train'):
    model.eval()
    iters = len(target_loader)
    target_iter = iter(target_loader)
    total_pred_all, total_pred_conf_all, image_paths, labels = [], [], [], []
    for i in range(iters):
        target_data, target_label, target_img_path = target_iter.next()
        target_data, target_label = target_data.to(device), target_label.to(device)

        # target_domain_label = torch.ones(target_data.shape[0]).long().to(device)

        pred = model(target_data)

        target_domain_pred =  torch.argmax(pred, dim=1)
        target_domain_conf = torch.softmax(pred, dim=1)

        total_pred_all.append(target_domain_pred)
        total_pred_conf_all.append(target_domain_conf)
        image_paths.append(target_img_path)
        labels.append(target_label)

    total_pred_all = torch.cat(total_pred_all, dim=0)
    total_pred_conf_all = torch.cat(total_pred_conf_all, dim=0)
    image_paths = torch.cat(image_paths, dim=0)
    labels = torch.cat(labels, dim=0)
    # write the pred and pred conf to txt file file
    with open(f'flir_{split}.txt', 'w') as f:
        for i in range(len(total_pred_all)):
            f.write(f'{image_paths[i]} {labels[i].item()} {total_pred_all[i].item()} {total_pred_conf_all[i].item()}\n')
    
def main():
    #update cfg
    cfg.merge_from_file('/home/rufael.marew/Documents/Academics/AI702/project/Transformers_for_Domain_adaptation/configs/train_disc.yaml')
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    source_train_dataset = CustomImageDataset(root=cfg.DATASETS.ROOT_TRAIN_DIR, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    source_test_dataset = CustomImageDataset(root=cfg.DATASETS.ROOT_TEST_DIR, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    target_train_dataset = CustomImageDataset(root=cfg.DATASETS.ROOT_TRAIN_DIR2, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    target_test_dataset = CustomImageDataset(root=cfg.DATASETS.ROOT_TEST_DIR2, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    # source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    # target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    source_train_loader = DataLoader(source_train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    source_test_loader = DataLoader(source_test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    target_train_loader = DataLoader(target_train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    #load model
    model = make_model(cfg, num_class=2, camera_num=0, view_num = 0)
    model = model.to(device)

    #load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    #load criterion
    criterion = torch.nn.CrossEntropyLoss()

    #train
    train(model, source_train_loader, target_train_loader, optimizer, criterion, device, epochs=100, print_freq=1000)

    #validate
    loss, pred = validate(model, source_test_loader, target_test_loader, criterion, device)
    print(f'Validation Loss {loss:.4f}\t validation Accuracy {pred:.4f}\t')

    #run infernce on target dataset
    infer_target(model, target_train_loader, device, split='train')
    infer_target(model, target_test_loader, device, split='test')

if __name__ == '__main__':

    main()