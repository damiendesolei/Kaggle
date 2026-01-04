# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 23:49:07 2025

@author: zrj-desktop
"""

# https://www.kaggle.com/code/jiazhuang/csiro-simple?scriptVersionId=286751968


import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from tqdm.auto import tqdm
tqdm.pandas()


# 环境设置

TRAIN = True  # submission时只用跑推理，设为 False

DEBUG = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Interactive'  # 交互式环境下会少跑一些epoch，用于快速跑通流程，方便调试
LOCAL = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == ''  # 用于在本地开发，觉得代码ok后通过 `kaggle k push` 命令提交到 kaggle 平台

if LOCAL:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    


# 训练设置

## CV
CV_STRATEGY = 'groupby_Sampling_Date'  # groupby_Sampling_Date
NFOLD = 6
KFOLD_SEED = 66

## Model
MODEL_NAME = 'convnext_small.fb_in22k_ft_in1k_384'

## Training Hyper Params
LR = 1e-5

## Model Output
OUTPUT_DIR = 'G:/kaggle/CSIRO_Biomass/models/convnext_small/'



############# Load Data #############

#DATA_ROOT = '../input/' if LOCAL else '/kaggle/input/csiro-biomass/'
DATA_ROOT = r'G:/kaggle/CSIRO_Biomass/data/'

train_df = pd.read_csv(f'{DATA_ROOT}train.csv')
train_df.head()

train_df[['sample_id_prefix', 'sample_id_suffix']] = train_df.sample_id.str.split('__', expand=True)

(train_df.sample_id_suffix == train_df.target_name).all()

cols = ['sample_id_prefix', 'image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
agg_train_df = train_df.groupby(cols).apply(lambda df: df.set_index('target_name').target)
agg_train_df.reset_index(inplace=True)
agg_train_df.columns.name = None

agg_train_df['image'] = agg_train_df.image_path.progress_apply(
    lambda path: Image.open(DATA_ROOT + path).convert('RGB')
)

agg_train_df.head()

agg_train_df['image_size'] = agg_train_df.image.apply(lambda x: x.size)
agg_train_df['image_size'].value_counts()


# Check columns
np.isclose(
    agg_train_df[['Dry_Green_g', 'Dry_Clover_g']].sum(axis=1),
    agg_train_df['GDM_g'],
    atol=1e-04
).mean()

np.isclose(
    agg_train_df[['GDM_g', 'Dry_Dead_g']].sum(axis=1),
    agg_train_df['Dry_Total_g'],
    atol=1e-04
).mean()


plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
agg_train_df.Dry_Green_g.plot(kind='hist')
_ = plt.title('Dry_Green_g')

plt.subplot(1, 3, 2)
agg_train_df.Dry_Clover_g.plot(kind='hist')
_ = plt.title('Dry_Clover_g')

plt.subplot(1, 3, 3)
agg_train_df.Dry_Dead_g.plot(kind='hist')
_ = plt.title('Dry_Dead_g')



############# CV Setup #############

agg_train_df['Sampling_Date_Month'] = agg_train_df.Sampling_Date.apply(lambda x: x.split('/')[1].strip())

# agg_train_df = agg_train_df.sort_index().sample(frac=1.0, random_state=31).copy()  # shuffle
# agg_train_df['idx'] = agg_train_df.index

# half_num = agg_train_df.shape[0] // 6
# print(half_num)

# head_df = agg_train_df.iloc[:half_num].reset_index(drop=True)
# tail_df = agg_train_df.iloc[half_num:].reset_index(drop=True)
# head_df.shape[0], tail_df.shape[0], len(set(head_df.idx) | set(tail_df.idx)) # cehck split

KFoldClass = StratifiedGroupKFold #if CV_STRATEGY == 'groupby_Sampling_Date' else KFold
kfold = KFoldClass(n_splits=NFOLD, shuffle=True, random_state=2025)

agg_train_df['fold'] = None
for i, (trn_idx, val_idx) in enumerate(kfold.split(agg_train_df.index, y=agg_train_df.State, groups=agg_train_df.Sampling_Date)):
    agg_train_df.loc[val_idx, 'fold'] = i
    
    trn_df = agg_train_df[agg_train_df.fold != i]
    val_df = agg_train_df[agg_train_df.fold == i]
    
    trn_months = sorted(int(x) for x in trn_df.Sampling_Date_Month.unique())
    val_months = sorted(int(x) for x in val_df.Sampling_Date_Month.unique())
    flag = val_df.Sampling_Date_Month.isin(trn_df.Sampling_Date_Month)
    print(f'trn({trn_df.shape[0]}) -> val({val_df.shape[0]}): {trn_months} -> {val_months} with {flag.mean()} of train in valid')
    
   
    

    

    
############# DataLoader ############

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class RegressionDataset(Dataset):
    def __init__(self, data, transform=None, vertical_split=True):
        self.data = data
        self.transform = transform
        self.vertical_split = vertical_split

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = item.image
        targets = [item['Dry_Green_g'], item['Dry_Clover_g'], item['Dry_Dead_g']]
        
        if self.vertical_split:
            # 垂直均分成左右两张图片
            width, height = image.size
            mid_point = width // 2
            left_image = image.crop((0, 0, mid_point, height))
            right_image = image.crop((mid_point, 0, width, height))
            
            if self.transform:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)
            
            return left_image, right_image, targets
        
        else:
            if self.transform:
                image = self.transform(image)

            return image, targets


def create_dataloader(data, target_image_size=(384, 384), batch_size=32, shuffle=True, aug=True, tta_transform=None):    
    if aug:
        transform = transforms.Compose([
            transforms.Resize(target_image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation([90, 90])], p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    else:
        if tta_transform:
            transform = transforms.Compose([
                transforms.Resize(target_image_size),
                tta_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(target_image_size),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    dataset = RegressionDataset(data, transform=transform)
    print('dataset size:', len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader


def get_tta_dataloaders(data, target_image_size, batch_size):
    res = []
    for transform in [None, transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation([90, 90])]:
        res.append(
            create_dataloader(data, target_image_size, batch_size, shuffle=False, aug=False, tta_transform=transform)
        )
    return res




############# Model #############
import timm
import torch
import torch.nn as nn

timm.list_models('*convnext_large*')
#MODEL_NAME = 'vit_large_patch16_dinov3_qkvb.lvd1689m'

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=3)
model.pretrained_cfg

TARGET_IMAGE_SIZE = model.pretrained_cfg['input_size'][1:]


class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        hidden = max(32, feat_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, feat_dim * 2)
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)
    
    
class MultiTargetRegressor(nn.Module):
    def __init__(self, model_name, pretrained=True, num_classes=3, dropout=0.0, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        
        self.film = FiLM(self.backbone.num_features)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features * 2, num_classes)
    
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    
    def forward(self, left_img, right_img):
        left_feat = self.backbone(left_img)
        right_feat = self.backbone(right_img)
        
        context = (left_feat + right_feat) / 2
        gamma, beta = self.film(context)
        
        left_feat_modulated = left_feat * (1 + gamma) + beta
        right_feat_modulated = right_feat * (1 + gamma) + beta
        
        combined = torch.cat([left_feat_modulated, right_feat_modulated], dim=1)
        
        features = self.dropout(combined)
        logits = self.head(features)
        return logits
    
    
    
    
    
    
############# Train #############
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


# ======== Weighted R2 ========
def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: shape (N, 5): Green/Clover/Dead/GDM/Total
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []
    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted_r2, r2_scores


def calc_metric(outputs, targets):
    '''
        outputs/targets: shape (N, 3): Green/Clover/Dead
    '''
    y_true = np.column_stack((
        targets,
        targets[:, :2].sum(axis=1),
        targets.sum(axis=1),
    ))
    
    y_pred = np.column_stack((
        outputs,
        outputs[:, :2].sum(axis=1),
        outputs.sum(axis=1),
    ))
    
    weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)
    return weighted_r2, r2_scores


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for left_images, right_images, targets in dataloader:  # unpack 3 values
        left_images = left_images.to(device)
        right_images = right_images.to(device)
        targets = torch.stack(targets).T.float().to(device)

        optimizer.zero_grad()
        outputs = model(left_images, right_images)  # pass both images
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for left_images, right_images, targets in dataloader:  # unpack 3 values
            left_images = left_images.to(device)
            right_images = right_images.to(device)
            targets = torch.stack(targets).T.float().to(device)

            outputs = model(left_images, right_images)  # pass both images
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    outputs = torch.cat(all_outputs).numpy()
    targets = torch.cat(all_targets).numpy()
    
    return total_loss / len(dataloader), outputs, targets
    
    # weighted_r2, r2_scores = calc_metric(outputs, targets)
    # return total_loss / len(dataloader), weighted_r2, r2_scores


def tta_validate(model, dataloaders, criterion, device):
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]
    
    all_loss = []
    all_outputs = []
    all_targets = []
    for dataloader in dataloaders:
        loss, outputs, targets = validate(model, dataloader, criterion, device)
        all_loss.append(loss)
        all_outputs.append(outputs)
        all_targets.append(targets)
    
    avg_loss = np.mean(all_loss)
    avg_outputs = np.mean(all_outputs, axis=0)
    avg_targets = np.mean(all_targets, axis=0)
    
    weighted_r2, r2_scores = calc_metric(avg_outputs, avg_targets)
    
    return avg_loss, weighted_r2, r2_scores


def train_fold(data, fold):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = 9
    lr = LR
    patience = 10
    num_epochs = 5 if DEBUG else 100
    warmup_ratio = 0.05

    # data
    train_loader = create_dataloader(data[data.fold != fold], TARGET_IMAGE_SIZE, batch_size, shuffle=True, aug=True)
    # val_loader = create_dataloader(data[data.fold == fold], TARGET_IMAGE_SIZE, batch_size, shuffle=False, aug=False)
    val_loaders = get_tta_dataloaders(data[data.fold == fold], TARGET_IMAGE_SIZE, batch_size)

    # model, loss, optimizer
    model = MultiTargetRegressor(MODEL_NAME, pretrained=True, num_classes=3) 
    model.to(device)

    criterion = nn.SmoothL1Loss()  # nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * num_training_steps)
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience//2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # Training loop
    history = []
    best_score = -float('inf')
    # best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        # val_loss, weighted_r2, r2_scores = validate(model, val_loader, criterion, device)
        val_loss, weighted_r2, r2_scores = tta_validate(model, val_loaders, criterion, device)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, weighted_r2: {weighted_r2:.4f}, lr: {optimizer.param_groups[0]['lr']}")
        
        history.append({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'weighted_r2': weighted_r2,
            'r2_scores': r2_scores,
        })
        
        # 早停
        if weighted_r2 > best_score:
        # if val_loss < best_loss:
            best_loss = val_loss
            best_score = weighted_r2
            epochs_without_improvement = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_mode_fold{fold}.pth')
        
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"早停: epoch={epoch}, {patience} 个 epoch 无改善")
            break

    print(f"\nTraining completed. Best weighted_r2: {best_score:.4f}")
    
    return history, best_score



if TRAIN:
    all_best_score = []

    for i in range(2 if DEBUG else NFOLD):
        print(f'### fold={i}')
        history, best_score = train_fold(agg_train_df, fold=i)
        all_best_score.append(best_score)
        history = pd.DataFrame(history)

        history.to_json(
            f'{OUTPUT_DIR}/history_fold{i}.jsonl',
            orient='records',
            lines=True,
            force_ascii=False,
        )

        # plot
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.title('LR')
        plt.plot(history.lr)

        plt.subplot(1, 3, 2)
        plt.title('Loss')
        plt.plot(history.train_loss, label='train')
        plt.plot(history.val_loss, label='val')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.title('weighted_r2')
        plt.plot(history.weighted_r2)
        plt.show()
    
    print('Avg CV:', np.mean(all_best_score))
    
    
    
    
    
    
############# Inference #############
import os
from pathlib import Path


def get_lastest_saved_models(model_root='/kaggle/input/csiro-simple-output/pytorch/default/'):    
    latest = 1
    for version in os.listdir(model_root):
        try:
            version = int(version)
        except:
            continue

        if version > latest:
            latest = version
    return f'{model_root}/trained_models/{latest}/'
    #return f'{model_root}/{latest}/trained_models/'

#SAVED_MODELS = './trained_models/' if TRAIN else get_lastest_saved_models('/kaggle/input/csiro-local-trained/pytorch/default/')

def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    all_outputs = []
    with torch.no_grad():
        for left_images, right_images, targets in dataloader:
            left_images = left_images.to(device)
            right_images = right_images.to(device)
    
            outputs = model(left_images, right_images)
            all_outputs.append(outputs.detach().cpu())
    
    outputs = torch.cat(all_outputs).numpy()
    return outputs


def tta_predict(model, dataloaders, device):
    all_outputs = []
    for dataloader in dataloaders:
        outputs = predict(model, dataloader, device)
        all_outputs.append(outputs)
    avg_outputs = np.mean(all_outputs, axis=0)
    return avg_outputs


def kfold_predict(dataloaders):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    all_preds = []
    for model_file in Path(SAVED_MODELS).glob('*.pth'):
        # model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=3)
        fold_idx = int(model_file.name.split('.')[0].split('fold')[1])
        if fold_idx >= NFOLD: continue
        print(model_file.name)
        model = MultiTargetRegressor(MODEL_NAME, pretrained=False, num_classes=3)
        model.load_state_dict(torch.load(model_file))

        preds = tta_predict(model, dataloaders, device)
        all_preds.append(preds)

    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


test_df = pd.read_csv(DATA_ROOT + 'test.csv')

test_df['target'] = 0.0
test_df[['sample_id_prefix', 'sample_id_suffix']] = test_df.sample_id.str.split('__', expand=True)

cols = ['sample_id_prefix', 'image_path']
agg_test_df = test_df.groupby(cols).apply(lambda df: df.set_index('target_name').target)
agg_test_df.reset_index(inplace=True)
agg_test_df.columns.name = None

agg_test_df['image'] = agg_test_df.image_path.progress_apply(
    lambda path: Image.open(DATA_ROOT + path).convert('RGB')
)

agg_test_df.head()



test_loader = get_tta_dataloaders(agg_test_df, TARGET_IMAGE_SIZE, 64)

SAVED_MODELS = 'G:/kaggle/CSIRO_Biomass/models/vit_large_patch16_dinov3_qkvb/1/'
preds = kfold_predict(test_loader)


agg_test_df[['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']] = preds
agg_test_df['GDM_g'] = agg_test_df.Dry_Green_g + agg_test_df.Dry_Clover_g
agg_test_df['Dry_Total_g'] = agg_test_df.GDM_g + agg_test_df.Dry_Dead_g

agg_test_df.head()



cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
sub_df = agg_test_df.set_index('sample_id_prefix')[cols].stack()
sub_df = sub_df.reset_index()
sub_df.columns = ['sample_id_prefix', 'target_name', 'target']

sub_df['sample_id'] = sub_df.sample_id_prefix + '__' + sub_df.target_name


cols = ['sample_id', 'target']
#sub_df[cols].to_csv('submission.csv', index=False)


#!head submission.csv