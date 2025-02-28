import torch
import torch.nn as nn
import os

from data import create_train_dataloader , create_val_dataloader

# Part One dataset loading 


train_dataloaders = {}
D1_train_path = "dataset/part_one_dataset/train_data/1_train_data.tar.pth"
D1_dataloader = create_train_dataloader(D1_train_path , flag = False)

train_dataloaders['D1_dataloader'] = D1_dataloader

for i in range(2 , 11):
    train_path = f"dataset/part_one_dataset/train_data/{i}_train_data.tar.pth"
    train_dataloaders[f"D{i}_train_dataloader"] = create_train_dataloader(train_path)
    
    
for i in range(1 , 11):
    train_path = f"dataset/part_two_dataset/train_data/{i}_train_data.tar.pth"
    train_dataloaders[f"D{i+10}_train_dataloader"] = create_train_dataloader(train_path)

for key, value in train_dataloaders.items():
    print(f"{key} created successfully")

val_dataloaders = {}

for i in range(1 , 11):
    val_path = f"dataset/part_one_dataset/eval_data/{i}_eval_data.tar.pth"
    val_dataloaders[f"D{i}_val_dataloader"] = create_val_dataloader(val_path)
    print(f"D{i}_val_dataloader loaded successfully")

for i in range(1 , 11):
    val_path = f"dataset/part_two_dataset/eval_data/{i}_eval_data.tar.pth"
    val_dataloaders[f"D{i+10}_val_dataloader"] = create_val_dataloader(val_path)
    print(f"D{i + 10}_val_dataloader loaded successfully")



    


