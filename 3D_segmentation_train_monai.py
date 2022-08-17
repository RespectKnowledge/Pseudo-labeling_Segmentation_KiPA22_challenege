# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:10:23 2022

@author: Administrateur
"""
#%%

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import shutil
import tempfile
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

### training and validation dataset
#path_train_volumes
###### create nnunet monaie for multiclass 3D segmentation problem
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
    Resized,RandShiftIntensityd,SpatialPadd
)
from monai.utils import first

### tranform the images
import os
from glob import glob
import shutil
from tqdm import tqdm
#import dicom2nifti
import numpy as np
import nibabel as nib
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism
set_determinism(seed=0)
import os
import nibabel as nib
import glob
#import SimpleITK as sitk
in_dir='/content/drive/MyDrive/KiPA2022/train_my/train'
#pathim=os.path.join(in_dir,'image')
#pathlabel=os.path.join(in_dir,'label')
path_train_volumes = sorted(glob.glob(os.path.join(in_dir, "image", "*.nii.gz")))
path_train_segmentation = sorted(glob.glob(os.path.join(in_dir, "label", "*.nii.gz")))
#image_keys: ["image"]
#all_keys: ["image", "label"]
train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]

#### dataset for training and validation

import pandas as pd
train_file=pd.read_csv('C:\\Users\\Usuario\\testmonaief\\train_fold0.csv')
in_dir='C:\\Users\\Usuario\\testmonaief\\train_my\\'
path_train_volumes=[]
path_train_segmentation=[]
for i in range(0,len(train_file)):
  pathtrain=train_file['PatientID'][i]
  path_train_volumes.append(os.path.join(in_dir,pathtrain))
  pathtrain.replace('image','label')
  path_train_segmentation.append(os.path.join(in_dir,pathtrain.replace('image','label')))


import pandas as pd
#in_dir='C:\\Users\\Usuario\\testmonaief\\train_my\\'
train_file=pd.read_csv('C:\\Users\\Usuario\\testmonaief\\valid_fold0.csv')
path_valid_volumes=[]
path_valid_segmentation=[]
for i in range(0,len(train_file)):
  pathtrain=train_file['PatientID'][i]
  pathf=os.path.join(in_dir,pathtrain)
  path_valid_volumes.append(pathf)
  pathtrain.replace('image','label')
  path_valid_segmentation.append(os.path.join(in_dir,pathtrain.replace('image','label')))

train_files = [{"image": image_name, 'label': label_name} 
               for image_name, label_name in zip(path_train_volumes, 
                                                 path_train_segmentation)]

valid_files = [{"image": image_name, 'label': label_name} 
               for image_name, label_name in zip(path_valid_volumes, 
                                                 path_valid_segmentation)]


#path_train_volumes
###### create nnunet monaie for multiclass 3D segmentation problem
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    Spacingd,
    Orientationd,
    SpatialPadd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandZoomd,
    CastToTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandFlipd,
    ToTensord,
    Resized,RandShiftIntensityd,SpatialPadd
)
from monai.utils import first
#image_keys: ["image"]
#all_keys: ["image", "label"]

generat_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(0.632813, 0.632813, 0.632813), mode=("bilinear", "nearest")),
        #Orientationd(keys=["image", "label"], axcodes="RAS"),
        #CropForegroundd(keys=['image','label'], source_key='image'),
        SpatialPadd(keys=["image", "label"], spatial_size=[128,128,128]),
        #Resized(keys=["image", "label"], spatial_size=[128,128,128]),
        ScaleIntensityRanged(keys=["image"], a_min=918.0, a_max=1396.0,b_min=0.0, b_max=1.0, clip=True,),
        RandCropByPosNegLabeld(  # crop with center in label>0 with proba pos / (neg + pos)
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128,128,128),
            pos=1,
            neg=0,  # never center in background voxels
            num_samples=4,
            image_key='image',  # for no restriction with image thresholding
            image_threshold=0,
        ), 
        RandGaussianNoised(keys="image", mean=0., std=0.1, prob=0.2),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.2,
          ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
            ),
        RandAdjustContrastd(  # same as Gamma in nnU-Net
            keys=["image"],
            gamma=(0.7, 1.5),
            prob=0.3,
          ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.7,
            max_zoom=1.5,
            mode=("trilinear",) * len(["image"]) + ("nearest",),
            align_corners=(True,) * len(["image"]) + (None,),
            prob=0.3,
          ),
          #RandRotated(
          #  keys=["image", "label"],
          #  range_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
           # range_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
           # range_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
           # mode=("bilinear",) * len(["image"]) + ("nearest",),
           # align_corners=(True,) * len(["image"]) + (None,),
            #padding_mode=("border", ) * len(["image", "label"]),
            #prob=0.3,
           # ),
        CastToTyped(keys=["image", "label"], dtype=(np.float32,) * len(["image"]) + (np.uint8,)),
        #RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
        #RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10), 
        #RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        #RandGaussianNoised(keys='image', prob=0.5),
        #NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ]
)


generat_ds = Dataset(data=train_files, transform=generat_transforms)
train_loader = DataLoader(generat_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

# #generat_loader = DataLoader(generat_ds, batch_size=4)
# for i,d in enumerate(train_loader):
#   print(d['image'].shape)
#   print(d['label'].shape)
#   break

# my_transform_org=Compose([LoadImaged(keys=["image", "label"]),AddChanneld(keys=["image", "label"]),
#                           ToTensord(keys=["image", "label"])])

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        #Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.632813, 0.632813, 0.632813),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=918.0, a_max=1396.0, b_min=0.0, b_max=1.0, clip=True
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        #Resized(keys=["image", "label"], spatial_size=[128,128,128]),
        #Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.632813, 0.632813, 0.632813),
            mode=("bilinear", "nearest"),
        ),
        #Resized(keys=["image", "label"], spatial_size=[128,128,128]),
        ScaleIntensityRanged(
            keys=["image"], a_min=918.0, a_max=1396.0, b_min=0.0, b_max=1.0, clip=True
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
valid_ds = Dataset(data=valid_files, transform=val_transforms)
val_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
#generat_loader = DataLoader(generat_ds, batch_size=4)
# for i,d in enumerate(valid_loader):
#   print(d['image'].shape)
#   print(d['label'].shape)
  #break


#train_loader = DataLoader(
   # train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
#)

#val_loader = DataLoader(
   # val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
#)
# generat_ds = Dataset(data=train_files, transform=val_transforms)
# generat_loader = DataLoader(generat_ds, batch_size=1)
# generat_patient = first(generat_loader)

# original_ds = Dataset(data=train_files, transform=my_transform_org)
# original_loader = DataLoader(original_ds, batch_size=1)
# original_patient = first(original_loader)

# my_transform_org=Compose([LoadImaged(keys=["image", "label"]),AddChanneld(keys=["image", "label"]),ToTensord(keys=["image", "label"])])
# generat_ds = Dataset(data=train_files, transform=generat_transforms)
# generat_loader = DataLoader(generat_ds, batch_size=4)
# for i,d in enumerate(generat_loader):
#   print(d['image'].shape)
#   print(d['label'].shape)
#   #break


############################ model defined ####################



import torch
from monai.networks.nets import DynUNet
#from task_params import deep_supr_num, patch_size, spacing

def get_kernels_strides():
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    #sizes, spacings = patch_size[192, 160, 64], spacing[0.79, 0.79, 1.6]
    sizes = (128,128,128)
    spacings = (0.632813, 0.632813, 0.632813)
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


# def get_network(properties, task_id, pretrain_path, checkpoint=None):
#     n_class = len(properties["labels"])
#     in_channels = len(properties["modality"])
#     kernels, strides = get_kernels_strides(task_id)

#     net = DynUNet(
#         spatial_dims=3,
#         in_channels=in_channels,
#         out_channels=n_class,
#         kernel_size=kernels,
#         strides=strides,
#         upsample_kernel_size=strides[1:],
#         norm_name="instance",
#         deep_supervision=True,
#         deep_supr_num=deep_supr_num[task_id],
#     )

#     if checkpoint is not None:
#         pretrain_path = os.path.join(pretrain_path, checkpoint)
#         if os.path.exists(pretrain_path):
#             net.load_state_dict(torch.load(pretrain_path))
#             print("pretrained checkpoint: {} loaded".format(pretrain_path))
#         else:
#             print("no pretrained checkpoint")
#     return net

def get_DynUNet(in_channels, n_class, device):
    """
    Return a 3D U-Net.
    :param config: config training parameters.
    :param in_channels: int. Number of input channels.
    :param n_class: int. Number of output classes.
    :param device: Device to use (cpu or gpu).
    :return:
    """
    #from src.networks.dynunet_compatibility import DynUNet  # DynUNet from MONAI 0.4.0+85.gaf1ffd6
    #DynUNet
    strides, kernels = [], []
    sizes = (128,128,128)
    spacings = (0.632813, 0.632813, 0.632813)
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supr_num=3,  # default is 1
        res_block=False,
    ).to(device)
    return net

in_channels=1
n_class=5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=get_DynUNet(in_channels, n_class, device)
#print(net)

#import torch
#inp=torch.rand(1,1,128,128,224).to('cpu')
#out=net1(inp)
#print(out.shape)



###################### typical pytorch training and validation loops


#### define models

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
root_dir='C:\\Users\\Usuario\\testmonaief\\model_save'
def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=5)
post_pred = AsDiscrete(argmax=True, to_onehot=5)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
    
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

# print(
#     f"train completed, best_metric: {dice_val_best:.4f} "
#     f"at iteration: {global_step_best}"
# )
# #%% output inferences
# plt.figure("train", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Iteration Average Loss")
# x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
# y = epoch_loss_values
# plt.xlabel("Iteration")
# plt.plot(x, y)
# plt.subplot(1, 2, 2)
# plt.title("Val Mean Dice")
# x = [eval_num * (i + 1) for i in range(len(metric_values))]
# y = metric_values
# plt.xlabel("Iteration")
# plt.plot(x, y)
# plt.show()

# case_num = 4
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
#     img = val_ds[case_num]["image"]
#     label = val_ds[case_num]["label"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_labels = torch.unsqueeze(label, 1).cuda()
#     val_outputs = sliding_window_inference(
#         val_inputs, (96, 96, 96), 4, model, overlap=0.8
#     )
#     plt.figure("check", (18, 6))
#     plt.subplot(1, 3, 1)
#     plt.title("image")
#     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.title("label")
#     plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
#     plt.subplot(1, 3, 3)
#     plt.title("output")
#     plt.imshow(
#         torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
#     )
#     plt.show()
    








