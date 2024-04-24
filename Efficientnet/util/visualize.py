import numpy as np
from typing import List,Dict
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
attribs_m = ['Roof Condition', 'Roof Material','Roof Style','Solar Panel', 'Tree Overhang', 'Swimming Pool']
item_2_label_lst = [
    {'Fair':0, 'Good':1, 'Poor':2,'Damaged':3},
    {'Metal':0, 'Poly':1, 'Shingle':2, 'Tile':3,'Ballasted':4,'Asphalt':5},
    {'Flat':0, 'Gabled':1, 'Hip':2, 'Mixed':3},
    {'No':0,'Yes':1},
    {'Low':0, 'Medium':1,'High':2, 'No':3},
    {'No':0, 'Yes':1,'Screened':2}]

def print_labels_and_preds(test_labels:List, pred_classes:List) -> None:
  for t,p in zip(test_labels, pred_classes):
    print(t)
    print(p)
    print("-"*150)
      
def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = [x.detach().cpu().numpy() for x in results['train_loss']]
    test_loss = [x.detach().cpu().numpy() for x in results['test_loss']]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig("results/loss_curves.png");
    
def plot_test_results(test_samples,test_data_custom:):
  # Plot predictions
  plt.figure(figsize=(9, 9))
  plt.suptitle(f'Test results: Output Format:\n{attribs_m}')
  nrows = 3
  ncols = 3
  for i, sample in enumerate(test_samples):
    # Create a subplot
    plt.subplot(nrows, ncols, i+1)
    targ_image_adjust = sample.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    # Plot the target image
    plt.imshow(targ_image_adjust)

    # Find the prediction label (in text form, e.g. "Sandal")
    pred_label = test_data_custom.decode_labels_onehot(pred_classes[i])

    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = test_data_custom.decode_labels_onehot(test_labels[i])

    # Create the title text of the plot
    title_text = f"Pred: {pred_label} \n Truth: {truth_label}"

    # Check for equality and change title colour accordingly
    if pred_label == truth_label:
        plt.title(title_text, fontsize=6, c="g") # green text if correct
    else:
        plt.title(title_text, fontsize=6, c="r") # red text if wrong
    plt.axis(False)
  #plt.savefig("results/test_results.png");

def plot_cm(cm:List, class_names:List(str),cls:str, ax):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Threshold for determining text color
    text_color_threshold = cm.max() / 2.  

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > text_color_threshold else 'black'  
            ax.text(j, i, format(cm[i, j], '.0f'), ha="center", va="center", color=color)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, 
           yticklabels=class_names,
           title= cls,
           ylabel='True label',
           xlabel='Predicted label');
    
def plot_confusion_matrices(y_pred_tensors:List(List),targets_tensors:List(List),
                            attribs:List(str) = attribs_m, item_2_label_lst:List(Dict)=item_2_label_lst):
  nrows,ncols = 2,3
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

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 9))
  plt.suptitle('Confusion Matrices')
  for i, (conf_mat,cls, class_names) in enumerate(zip(confmats,attribs, cls_lst)):
    row = i // ncols
    col = i % ncols
    ax = axes[row, col]  
    plot_cm(conf_mat, class_names,cls, ax) 
  plt.tight_layout()
  plt.show()
  #plt.savefig("results/cm_me.png");
