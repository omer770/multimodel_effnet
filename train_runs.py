import os
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights

from multimodel_effnet.Efficientnet.model import multimodel_effnet_arc
from multimodel_effnet.Efficientnet.util.dataset import display_random_images,CustomImageDataset
from multimodel_effnet.Efficientnet.util.metrics_utils import cross_entropy_loss_embedded,accuracy_embedded
from multimodel_effnet.Efficientnet.util.train_utils import train
from multimodel_effnet.Efficientnet.util.utils import save_model
from multimodel_effnet.Efficientnet.util.visualize import plot_loss_curves

device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_device(device)
print("device: ",device)
base_path = Path(folder_path)
#csv_path = base_path/'multimodel_effnet'/'data'/ csv_name
train_csv_path = base_path/'multimodel_effnet'/'data'/ test_csv_name
test_csv_path = base_path/'multimodel_effnet'/'data'/ test_csv_name
image_directory = '/content/multimodel_effnet/data/Datasets2'
weights_Dir = Path('/content/drive/MyDrive/Colab_zip/multimodel_effnet/weights')
weights_Dir.mkdir(parents=True, exist_ok=True)
attribs_m = ['Roof Condition', 'Roof Material','Roof Style','Solar Panel', 'Tree Overhang', 'Swimming Pool']
item_2_label_lst = [
    {'Fair':0, 'Good':1, 'Poor':2,'Damaged':3},
    {'Metal':0, 'Poly':1, 'Shingle':2, 'Tile':3,'Ballasted':4,'Asphalt':5},
    {'Flat':0, 'Gabled':1, 'Hip':2, 'Mixed':3},
    {'No':0,'Yes':1},
    {'Low':0, 'Medium':1,'High':2, 'No':3},
    {'No':0, 'Yes':1,'Screened':2}]
label_2_item_lst = [
    {0: 'Fair', 1: 'Good', 2: 'Poor', 3:'Damaged'},
    {0: 'Metal', 1: 'Poly', 2: 'Shingle', 3: 'Tile', 4: 'Ballasted', 5:'Asphalt'},
    {0: 'Flat', 1: 'Gabled', 2: 'Hip', 3: 'Mixed'},
    {0: 'No', 1:'Yes'},
    {0: 'Low', 1: 'Medium', 2:'High', 3: 'No'},
    {0: 'No', 1: 'Yes', 2:'Screened'}]
#print(df.head())

weights = EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
effnet = efficientnet_b0
model = multimodel_effnet_arc.effnet_multimodel(model_name='b1',effnet=effnet, weights =weights)
# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.effnet.parameters():
    param.requires_grad = False


# Augment train data
train_transforms = transforms.Compose([
    transforms.CenterCrop(768),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.CenterCrop(768),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
train_data_custom = CustomImageDataset(annotations_file=train_csv_path,
                                      img_dir=image_directory,
                                      attribs=attribs_m,
                                      item_2_label_lst =item_2_label_lst,
                                      label_2_item_lst=label_2_item_lst,
                                      transform=train_transforms)
test_data_custom = CustomImageDataset(annotations_file=test_csv_path,
                                      img_dir=image_directory,
                                      attribs=attribs_m,
                                      item_2_label_lst =item_2_label_lst,
                                      label_2_item_lst=label_2_item_lst,
                                      transform=test_transforms)
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

train_dataloader = DataLoader(
      train_data_custom,
      batch_size=BATCH_SIZE,
      #shuffle=True,
      #num_workers=NUM_WORKERS,
      pin_memory=True,
      generator=torch.Generator(device=device)
  )
test_dataloader = DataLoader(
      test_data_custom,
      batch_size=BATCH_SIZE,
      #shuffle=False,
      #num_workers=NUM_WORKERS,
      pin_memory=True,
      generator=torch.Generator(device=device)
  )

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

result = train(model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=cross_entropy_loss_embedded,
      acc_fn= accuracy_embedded,
      epochs=5,
      device=device)

if __name__=="__main__":
  result = train(model=model,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=cross_entropy_loss_embedded,
      acc_fn= accuracy_embedded,
      epochs=5,
      device=device)
  return result

