import torch
from typing import List
from pprint import pprint
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def cross_entropy_loss_embedded(list_pred:List, list_y:List) -> float:
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0

    for y_pred, y_true in zip(list_pred, list_y):
        target = y_true.argmax(dim=1)  # Targets are indices for each item in the batch
        loss = loss_fn(y_pred, target)
        total_loss += loss

    return total_loss / len(list_pred)  # Average over the number of attributes

def accuracy_embedded(list_pred:List, list_y:List) -> float:
    total_correct = 0
    total_samples = 0

    for y_pred, y_true in zip(list_pred, list_y):
        preds = y_pred.argmax(dim=1)
        targets = y_true.argmax(dim=1)
        total_samples += preds.size(0)  # Increment sample count based on batch size
        total_correct += (preds == targets).sum().item()

    return total_correct / total_samples

def calculate_ConfusionMatrices(y_pred_tensors:List,targets_tensors:List,
                            attribs:List = attribs_m, item_2_label_lst:List=item_2_label_lst):
  [y_pred_tensor_c,y_pred_tensor_m,y_pred_tensor_y,
   y_pred_tensor_s,y_pred_tensor_t,y_pred_tensor_p] = y_pred_tensors
  [test_data_targets_tensor_c,test_data_targets_tensor_m,
   test_data_targets_tensor_y,test_data_targets_tensor_s,
   test_data_targets_tensor_t,test_data_targets_tensor_p] = targets_tensors
  confmat_c = ConfusionMatrix(num_classes=len(item_2_label_lst[0]), task='multiclass')
  confmat_m = ConfusionMatrix(num_classes=len(item_2_label_lst[1]), task='multiclass')
  confmat_y = ConfusionMatrix(num_classes=len(item_2_label_lst[2]), task='multiclass')
  confmat_s = ConfusionMatrix(num_classes=len(item_2_label_lst[3]), task='multiclass')
  confmat_t = ConfusionMatrix(num_classes=len(item_2_label_lst[4]), task='multiclass')
  confmat_p = ConfusionMatrix(num_classes=len(item_2_label_lst[5]), task='multiclass')

  confmat_tensor_c = confmat_c(preds=y_pred_tensor_c,target=test_data_targets_tensor_t)
  confmat_tensor_m = confmat_m(preds=y_pred_tensor_m,target=test_data_targets_tensor_m)
  confmat_tensor_y = confmat_y(preds=y_pred_tensor_y,target=test_data_targets_tensor_y)
  confmat_tensor_s = confmat_s(preds=y_pred_tensor_s,target=test_data_targets_tensor_s)
  confmat_tensor_t = confmat_t(preds=y_pred_tensor_t,target=test_data_targets_tensor_t)
  confmat_tensor_p = confmat_p(preds=y_pred_tensor_p,target=test_data_targets_tensor_p)
  confmats = [
    confmat_tensor_c.cpu().numpy(),
    confmat_tensor_m.cpu().numpy(),
    confmat_tensor_y.cpu().numpy(),
    confmat_tensor_s.cpu().numpy(),
    confmat_tensor_t.cpu().numpy(),
    confmat_tensor_p.cpu().numpy()]
  cls_lst = [list(item.keys()) for item in item_2_label_lst]

  for i, (conf_mat,cls, class_names) in enumerate(zip(confmats,attribs, cls_lst)):
    print("Attribute: ",end=' ')
    pprint(cls)
    print("Classes: ",end=' ')
    pprint(class_names)
    print("Confusion Matrix: ")
    pprint(conf_mat)
    print('-'*75) 
