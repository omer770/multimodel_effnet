import torch
from tqdm import tqdm
from pathlib import Path
from typing  import  Tuple, Dict, List
from IPython.display import clear_output
device  = 'cuda' if torch.cuda.is_available() else 'cpu'

weights_Dir = Path('/content/drive/MyDrive/Colab_zip/multimodel_effnet/weights')
weights_Dir.mkdir(parents=True, exist_ok=True)


'''
filePaths = [file for file in weights_Dir.iterdir() if file.name.startswith('me_model_weights')]
try:
  latest_weigths = str(filePaths[-1])
  #model.load_state_dict(torch.load(latest_weigths,map_location= device))
  print("choosen weights: ",latest_weigths)
  times = str(int(latest_weigths.split('_')[-2])+1).zfill(2)
except:
  latest_weigths= None
  print("choosen weights: ",latest_weigths)
  times = '00'
'''
# Dataset paths
#images_files=sorted(os.listdir("data/images"))

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               acc_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
              verbose: int = 0) -> Tuple[float, float]:

    # Put model in train mode
    model.to(device)
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0
    train_acc = 0
    # Loop through data loader data batches
    print("\nTotal batches: ",len(dataloader))
    for batch, (X, (y_c,y_m,y_y,y_s,y_t,y_p)) in enumerate(dataloader):
        # Send data to target device
        #print(f"Train - batch: {batch}, X: {X.shape}, y: {len(y_c)},...,{len(y_m)} ")
        X = X.to(device)
        y_c,y_m,y_y,y_s,y_t,y_p = y_c.to(device),y_m.to(device),y_y.to(device),y_s.to(device),y_t.to(device),y_p.to(device)

        # 1. Forward pass
        y_pred_c,y_pred_m,y_pred_y,y_pred_s,y_pred_t,y_pred_p = model(X)

        # 2. Calculate  and accumulate loss
        #print(y_pred,y)
        #loss = loss_fn(y_pred, y)
        loss = loss_fn([y_pred_c,y_pred_m,y_pred_y,y_pred_s,y_pred_t,y_pred_p], [y_c,y_m,y_y,y_s,y_t,y_p])

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward

        loss.backward()
          # 5. Optimizer step
        optimizer.step()
        #print(f'Loss: {loss}')
        train_loss+= loss

        # Calculate and accumulate accuracy metric across all batches
        acc = acc_fn([y_pred_c,y_pred_m,y_pred_y,y_pred_s,y_pred_t,y_pred_p], [y_c,y_m,y_y,y_s,y_t,y_p])
        train_acc += acc
        if verbose ==1:
          print(f"Train - batch: {batch}, X: {X.shape}, y: {len(y_c)},...,{len(y_m)} Loss: {loss:.2f}, Accuracy: {acc:.2f} ")
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    #print(f'Train: Loss: {train_loss},Accuracy: {train_acc}')
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              acc_fn: torch.nn.Module,
              device: torch.device,
             verbose:int = 0) -> Tuple[float, float]:

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss = 0
    test_acc = 0
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, (y_c,y_m,y_y,y_s,y_t,y_p)) in enumerate(dataloader):
            # Send data to target device
            #print(f"Test - batch: {batch}, X: {X.shape}, y: {len(y_c)},...,{len(y_m)} ")
            X = X.to(device)
            y_c,y_m,y_y,y_s,y_t,y_p = y_c.to(device),y_m.to(device),y_y.to(device),y_s.to(device),y_t.to(device),y_p.to(device)

            # 1. Forward pass
            test_pred_c,test_pred_m,test_pred_y,test_pred_s,test_pred_t,test_pred_p = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn([test_pred_c,test_pred_m,test_pred_y,test_pred_s,test_pred_t,test_pred_p], [y_c,y_m,y_y,y_s,y_t,y_p])
            test_loss += loss

            acc= acc_fn([test_pred_c,test_pred_m,test_pred_y,test_pred_s,test_pred_t,test_pred_p], [y_c,y_m,y_y,y_s,y_t,y_p])
            # Calculate and accumulate accuracy
            test_acc += acc
            if verbose ==1:
              print(f"Test - batch: {batch}, X: {X.shape}, y: {len(y_c)},...,{len(y_m)} Loss: {loss:.2f}, Accuracy: {acc:.2f} ")
    # Adjust metrics to get average loss and accuracy per batch

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    #print(f'Test: Loss: {test_loss},Accuracy: {test_acc}')
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          acc_fn: torch.nn.Module,
          epochs: int,
          save_path:str,
          device: torch.device,
          latest_weigths:str = None,
          save_epoch:int=5
         verbose:int = 0) -> Dict[str, List]:

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    if latest_weigths:
      print("choosen weights: ",latest_weigths)
      times = str(int(latest_weigths.split('_')[-2])+1).zfill(2)
      model.load_state_dict(torch.load(latest_weigths,map_location= device))
    else:
      print("choosen weights: ",latest_weigths)
      times = '00'
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          acc_fn= acc_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          verbose = verbose)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          acc_fn = acc_fn,
          device=device,
          verbose = verbose)
        clear_output(wait=True)
        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if ((epoch+1)%save_epoch)==0:
            # Save the model's weights after each epoch
            torch.save(model.state_dict(), f"{str(save_path)}/me_model_weights_{times}_{str(epoch+1).zfill(3)}.pth")
            print(f"Model weights saved to {str(save_path)}/me_model_weights_{times}_{str(epoch+1).zfill(3)}.pth")
    # Return the filled results at the end of the epochs
    return results
