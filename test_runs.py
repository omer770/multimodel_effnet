import os
import torch
import random
import torchvision 
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from multimodel_effnet.Efficientnet.model import multimodel_effnet_arc
from multimodel_effnet.Efficientnet.util.dataset import CustomImageDataset
from multimodel_effnet.Efficientnet.util.metrics_utils import calculate_ConfusionMatrices
from multimodel_effnet.Efficientnet.util.visualize import plot_confusion_matrices

weights_Dir = Path('/content/drive/MyDrive/Colab_zip/multimodel_effnet/weights')
weights_Dir.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
filePaths = [file for file in weights_Dir.iterdir() if file.name.startswith('me_model_weights')]

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

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
  pred_probs_c,pred_probs_m,pred_probs_y,pred_probs_s,pred_probs_t,pred_probs_p = [],[],[],[],[],[]
  model.eval()
  with torch.inference_mode():
    for sample in data:
      # Prepare sample
      sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

      # Forward pass (model outputs raw logit)
      test_pred_c,test_pred_m,test_pred_y,test_pred_s,test_pred_t,test_pred_p = model(sample)

      # Get prediction probability (logit -> prediction probability)
      pred_prob_c = torch.softmax(test_pred_c.squeeze(), dim=0)
      pred_prob_m = torch.softmax(test_pred_m.squeeze(), dim=0)
      pred_prob_y = torch.softmax(test_pred_y.squeeze(), dim=0)
      pred_prob_s = torch.softmax(test_pred_s.squeeze(), dim=0)
      pred_prob_t = torch.softmax(test_pred_t.squeeze(), dim=0)
      pred_prob_p = torch.softmax(test_pred_p.squeeze(), dim=0)

      pred_probs_c.append(pred_prob_c.cpu())
      pred_probs_m.append(pred_prob_m.cpu())
      pred_probs_y.append(pred_prob_y.cpu())
      pred_probs_s.append(pred_prob_s.cpu())
      pred_probs_t.append(pred_prob_t.cpu())
      pred_probs_p.append(pred_prob_p.cpu())

    pred_classes_c = get_ohe_of_tensor(torch.stack(pred_probs_c))
    pred_classes_m = get_ohe_of_tensor(torch.stack(pred_probs_m))
    pred_classes_y = get_ohe_of_tensor(torch.stack(pred_probs_y))
    pred_classes_s = get_ohe_of_tensor(torch.stack(pred_probs_s))
    pred_classes_t = get_ohe_of_tensor(torch.stack(pred_probs_t))
    pred_classes_p = get_ohe_of_tensor(torch.stack(pred_probs_p))
  pred_classes= []
  for i in range(len(pred_classes_c)):
    pred_classes.append([
      pred_classes_c[i],pred_classes_m[i],pred_classes_y[i],
      pred_classes_s[i],pred_classes_t[i],pred_classes_p[i]])

  # Stack the pred_probs to turn list into a tensor
  return pred_classes

def get_ohe_of_tensor(pred_probs):
  zl = torch.zeros_like(pred_probs)
  for p in range(len(pred_probs)):
    pp = pred_probs[p].argmax(dim=0)
    #print(pp)
    zl[p][pp] = 1.
  return zl

def model_test(test_dataloader:torch.utils.data.DataLoader, model: torch.nn.Module,latest_weigths = None , device: torch.device = device):
  print("choosen weights: ",latest_weigths)
  if latest_weigths:
    model.load_state_dict(torch.load(latest_weigths,map_location= device))
  pred_probs_c,pred_probs_m,pred_probs_y,pred_probs_s,pred_probs_t,pred_probs_p = [],[],[],[],[],[]
  test_data_targets_c,test_data_targets_m,test_data_targets_y,test_data_targets_s,test_data_targets_t,test_data_targets_p = [],[],[],[],[],[]
  model.eval()
  with torch.inference_mode():
    for X, (y_c,y_m,y_y,y_s,y_t,y_p) in tqdm(test_dataloader, desc="Making predictions"):
      # Send data and targets to target device
      X = X.to(device)
      y_c,y_m,y_y,y_s,y_t,y_p = y_c.to(device),y_m.to(device),y_y.to(device),y_s.to(device),y_t.to(device),y_p.to(device)
      test_data_targets_c.append(y_c.argmax(dim=1).cpu())
      test_data_targets_m.append(y_m.argmax(dim=1).cpu())
      test_data_targets_y.append(y_y.argmax(dim=1).cpu())
      test_data_targets_s.append(y_s.argmax(dim=1).cpu())
      test_data_targets_t.append(y_t.argmax(dim=1).cpu())
      test_data_targets_p.append(y_p.argmax(dim=1).cpu())
      # 1. Forward pass
      test_pred_c,test_pred_m,test_pred_y,test_pred_s,test_pred_t,test_pred_p = model(X)

      pred_prob_c = torch.softmax(test_pred_c, dim=1).argmax(dim=1)
      pred_prob_m = torch.softmax(test_pred_m, dim=1).argmax(dim=1)
      pred_prob_y = torch.softmax(test_pred_y, dim=1).argmax(dim=1)
      pred_prob_s = torch.softmax(test_pred_s, dim=1).argmax(dim=1)
      pred_prob_t = torch.softmax(test_pred_t, dim=1).argmax(dim=1)
      pred_prob_p = torch.softmax(test_pred_p, dim=1).argmax(dim=1)

      pred_probs_c.append(pred_prob_c.cpu())
      pred_probs_m.append(pred_prob_m.cpu())
      pred_probs_y.append(pred_prob_y.cpu())
      pred_probs_s.append(pred_prob_s.cpu())
      pred_probs_t.append(pred_prob_t.cpu())
      pred_probs_p.append(pred_prob_p.cpu())
  test_data_targets_tensor_c = torch.cat(test_data_targets_c)
  test_data_targets_tensor_m = torch.cat(test_data_targets_m)
  test_data_targets_tensor_y = torch.cat(test_data_targets_y)
  test_data_targets_tensor_s = torch.cat(test_data_targets_s)
  test_data_targets_tensor_t = torch.cat(test_data_targets_t)
  test_data_targets_tensor_p = torch.cat(test_data_targets_p)
  # Concatenate list of predictions into a tensor
  y_pred_tensor_c = torch.cat(pred_probs_c)
  y_pred_tensor_m = torch.cat(pred_probs_m)
  y_pred_tensor_y = torch.cat(pred_probs_y)
  y_pred_tensor_s = torch.cat(pred_probs_s)
  y_pred_tensor_t = torch.cat(pred_probs_t)
  y_pred_tensor_p = torch.cat(pred_probs_p)

  y_pred_tensors = [y_pred_tensor_c,y_pred_tensor_m,y_pred_tensor_y,
  y_pred_tensor_s,y_pred_tensor_t,y_pred_tensor_p]
  targets_tensors = [test_data_targets_tensor_c,test_data_targets_tensor_m,
   test_data_targets_tensor_y,test_data_targets_tensor_s,
   test_data_targets_tensor_t,test_data_targets_tensor_p]
  return targets_tensors,y_pred_tensors

if __name__ == '__main__':
  BATCH_SIZE= 32
  NUM_WORKERS = os.cpu_count()
  image_directory = '/content/multimodel_effnet/data/Datasets2'
  test_csv_path = '/content/multimodel_effnet/data/Twia_with_id_new_balanced_test.csv'
  weights = EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
  effnet = efficientnet_b0
  model = multimodel_effnet_arc.effnet_multimodel(model_name='b1',effnet=effnet, weights =weights)
  # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
  for param in model.effnet.parameters():
      param.requires_grad = False
  try:
    latest_weigths = str(filePaths[-1])
    #model.load_state_dict(torch.load(latest_weigths,map_location= device))
    #print("choosen weights: ",latest_weigths)
  except:
    latest_weigths= None
    #print("choosen weights: ",latest_weigths)
  test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(768),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         #std=[0.229, 0.224, 0.225]),
  ])

  test_data_custom = CustomImageDataset(annotations_file=test_csv_path,
                                      img_dir=image_directory,
                                      attribs=attribs_m,
                                      item_2_label_lst =item_2_label_lst,
                                      label_2_item_lst=label_2_item_lst,
                                      #device=device,
                                      transform=test_transforms)
  test_dataloader = DataLoader(
      test_data_custom,
      batch_size=BATCH_SIZE,
      #shuffle=False,
      num_workers=NUM_WORKERS,
      pin_memory=True,
      generator=torch.Generator(device='cpu')
  )
  
  targets_tensors,y_pred_tensors = model_test( test_dataloader = test_dataloader, model = model,latest_weigths=latest_weigths, device = device)
  calculate_ConfusionMatrices(y_pred_tensors,targets_tensors)
  plot_confusion_matrices(y_pred_tensors,targets_tensors)
