import os
import cv2
import torch
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
from typing  import  Tuple, Dict, List

def create_balanced_df(df,cls,k):
  def sampling_k_elements(group, k=k):
    if len(group) < k:
        return group
    return group.sample(k)
  return df.groupby(cls).apply(sampling_k_elements).reset_index(drop=True)
  
class CustomImageDataset(torch.utils.data.dataset.Dataset):
  def __init__(self, img_dir: str,attribs:List,
               item_2_label_lst:List,label_2_item_lst:List,
               annotations_file: str = None,
               annotations_df: pd.DataFrame = None,
               device:torch.device ='cpu', transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.attribs = attribs
    self.item_2_label_lst = item_2_label_lst
    self.label_2_item_lst = label_2_item_lst
    self.device = device
    if annotations_file:
      self.label_df = pd.read_csv(annotations_file)
    if annotations_df is not None:
      self.label_df = annotations_df.copy()

  def __len__(self):
    return len(self.label_df)

  def __getitem__(self, index):
    image = self.load_image(index)
    label_val = list(self.label_df.loc[index, self.attribs])
    if self.transform:
      image = self.transform(image).cpu()
    ecd_labels = self.encode_labels_onehot(label_val)
    return image,ecd_labels

  def load_image(self, index: int) -> Image.Image:
      "Opens an image via a path and returns it."
      image_path = os.path.join(self.img_dir, self.label_df.loc[index, 'Image_ID'])
      #img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
      #Image.fromarray(img)
      return Image.open(image_path).convert("RGB")

  def encode_labels_onehot(self,labels):
    """Encodes labels as one-hot vectors using PyTorch tensors.

    Args:
        labels: A list of strings representing labels.
        item_2_label_lst: A list of dictionaries mapping text labels to integer indices.

    Returns:
        A list of one-hot encoded vectors (as PyTorch tensors).
    """
    encoded_labels = []
    for label_index, label_dict in enumerate(self.item_2_label_lst):
        label = labels[label_index]
        encoded_label_index = label_dict.get(label)
        if encoded_label_index is None:
            raise ValueError(f"Label '{label}' not found in mapping.")

        num_classes = len(label_dict)
        onehot_vector = torch.zeros(num_classes).cpu()
        onehot_vector[encoded_label_index] = 1
        encoded_labels.append(onehot_vector)

    return encoded_labels

  def decode_labels_onehot(self, onehot_labels):
    """Decodes one-hot encoded labels using PyTorch tensors.

    Args:
        onehot_labels: A list of one-hot encoded vectors (as PyTorch tensors).
        label_2_item_lst: A list of dictionaries mapping integer indices to text labels.

    Returns:
        A list of decoded labels (strings).
    """
    decoded_labels = []
    for label_index, onehot_vector in enumerate(onehot_labels):
        encoded_label_index = onehot_vector.argmax()  # PyTorch's argmax
        label_dict = self.label_2_item_lst[label_index]
        decoded_label = label_dict.get(encoded_label_index.item())  # Convert tensor index to int
        decoded_labels.append(decoded_label)

    return decoded_labels
class WeightedSampler(Sampler):
    def __init__(self, dataset, weights_strategy='image-level',power=1):
        self.dataset = dataset
        self.weights_strategy = weights_strategy
        self.power = power
        self.indices = list(range(len(dataset)))
        self.weights = self.calculate_weights(strategy=weights_strategy)
          
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
      #print("Sampled Indices:", random.choices(self.indices, weights=self.weights, k=len(self.dataset), replace=True))  
      return iter(random.choices(self.indices, weights=self.weights, k=len(self.dataset)))

    def calculate_weights(self, strategy='class-level'):
        if strategy == 'image-level':
          self.dict_att_wgts = {}
          for att in self.dataset.attribs:
            all_labels_for_attribute = self.dataset.label_df[att] 
            all_class_label = np.unique(all_labels_for_attribute, return_counts=True)[0]
            all_class_counts = np.unique(all_labels_for_attribute, return_counts=True)[1]
            max_count = all_class_counts.max()
            min_count = all_class_counts.min()
            freq_ratio = max_count / min_count  
            class_weights = {}
            for cls, count in zip(all_class_label,all_class_counts):
              #(5.0/len(all_class_label))*
              class_weights[cls] = (freq_ratio / count)**self.power
            self.dict_att_wgts[att] = class_weights
          weights = []
          for idx in self.indices:
              image_level_weight = 0
              label_val = self.dataset.label_df.loc[idx, self.dataset.attribs]
              for att in self.dataset.attribs:
                class_weight = self.dict_att_wgts[att][label_val[att]]
                image_level_weight += class_weight
              #print(f"{idx}-{self.dataset.label_df.loc[idx,'Image_ID']}:{image_level_weight}") 
              weights.append(image_level_weight)
        else:
            raise ValueError(f"Invalid weights strategy: {strategy}")
        #print("weights",weights)
        return weights
if __name__ == "__main__":
  pass
