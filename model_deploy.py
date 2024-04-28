import os
import torch
from pathlib import Path
import pathlib
import torch
import pandas as pd
from PIL import Image
from timeit import default_timer as timer 
from tqdm.auto import tqdm
from typing import List, Tuple, Dict
from transformers import pipeline
import gradio as gr
import torchvision

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

def pred_label_2_classes (pred_label,label_2_item_lst=label_2_item_lst):
  pred_class = [label_2_item_lst[i][label] for i,label in enumerate(pred_label)]
  return pred_class

def pred_logits_to_labels(pred_logits):
  pred_c,pred_m,pred_y,pred_s,pred_t,pred_p = pred_logits
  conf_c = torch.softmax(pred_c, dim=1)
  conf_m = torch.softmax(pred_m, dim=1)
  conf_y = torch.softmax(pred_y, dim=1)
  conf_s = torch.softmax(pred_s, dim=1)
  conf_t = torch.softmax(pred_t, dim=1)
  conf_p = torch.softmax(pred_p, dim=1)
  pred_label_c = conf_c.argmax(dim=1).cpu().item()
  pred_label_m = conf_m.argmax(dim=1).cpu().item()
  pred_label_y = conf_y.argmax(dim=1).cpu().item()
  pred_label_s = conf_s.argmax(dim=1).cpu().item()
  pred_label_t = conf_t.argmax(dim=1).cpu().item()
  pred_label_p = conf_p.argmax(dim=1).cpu().item()
  pred_label = [pred_label_c,pred_label_m,pred_label_y,pred_label_s,pred_label_t,pred_label_p]
  return pred_label

def pred_and_store(model: torch.nn.Module,
                   transform: torchvision.transforms, 
                   annotations_file:str, 
                   image_dir: str ,
                   get_scores: bool = False,
                   attribs: List[str] = attribs_m,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
    df_test = pd.read_csv(annotations_file)
    # 2. Create an empty list to store prediction dictionaires
    pred_list = []
    
    # 3. Loop through target paths
    for indx in tqdm(df_test.index):
        
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict["image_path"] = os.path.join(image_dir,df_test.loc[indx,'Image_ID'])
        pred_dict["classes"] = list(df_test.loc[indx,attribs_m].values)
        
        # 6. Start the prediction timer
        start_time = timer()
        
        # 7. Open image path
        img = Image.open(pred_dict["image_path"]).convert("RGB")
        
        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device) 
        
        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()
        
        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            test_pred_c,test_pred_m,test_pred_y,test_pred_s,test_pred_t,test_pred_p = model(transformed_image) # perform inference on target sample 
            pred_label_c = torch.softmax(test_pred_c, dim=1).argmax(dim=1).cpu().item()
            pred_label_m = torch.softmax(test_pred_m, dim=1).argmax(dim=1).cpu().item()
            pred_label_y = torch.softmax(test_pred_y, dim=1).argmax(dim=1).cpu().item()
            pred_label_s = torch.softmax(test_pred_s, dim=1).argmax(dim=1).cpu().item()
            pred_label_t = torch.softmax(test_pred_t, dim=1).argmax(dim=1).cpu().item()
            pred_label_p = torch.softmax(test_pred_p, dim=1).argmax(dim=1).cpu().item()

            pred_label = [pred_label_c,pred_label_m,pred_label_y,pred_label_s,pred_label_t,pred_label_p]

            pred_classes = pred_label_2_classes(pred_label) # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
            #pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_classes"] = pred_classes
            
            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict["correct"] = pred_dict["classes"] == pred_dict["pred_classes"]
        #pred_dict["score"] = 0
        if get_scores:
          count = 0
          for act,pred in zip(pred_dict["classes"],pred_dict["pred_classes"]):
            if act == pred : count+=1
          pred_dict["score"] = count
        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)
    
    # 15. Return list of prediction dictionaries
    return pred_list

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    
    # Transform the target image and add a batch dimension
    img = test_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_c,pred_m,pred_y,pred_s,pred_t,pred_p = model(img)
        conf_c = torch.softmax(pred_c, dim=1)
        conf_m = torch.softmax(pred_m, dim=1)
        conf_y = torch.softmax(pred_y, dim=1)
        conf_s = torch.softmax(pred_s, dim=1)
        conf_t = torch.softmax(pred_t, dim=1)
        conf_p = torch.softmax(pred_p, dim=1)
        pred_label_c = conf_c.argmax(dim=1).cpu().item()
        pred_label_m = conf_m.argmax(dim=1).cpu().item()
        pred_label_y = conf_y.argmax(dim=1).cpu().item()
        pred_label_s = conf_s.argmax(dim=1).cpu().item()
        pred_label_t = conf_t.argmax(dim=1).cpu().item()
        pred_label_p = conf_p.argmax(dim=1).cpu().item()

        pred_label = [pred_label_c,pred_label_m,pred_label_y,pred_label_s,pred_label_t,pred_label_p]

        pred_classes = pred_label_2_classes(pred_label)
        conf_classes = [round(conf_c.max().cpu().item(),2),round(conf_m.max().cpu().item(),2),
                        round(conf_y.max().cpu().item(),2),round(conf_s.max().cpu().item(),2),
                        round(conf_t.max().cpu().item(),2),round(conf_p.max().cpu().item(),2)]
    # Calculate the prediction time
    #pred_classes_dict = {att:(pred_classes[i],conf_classes[i]) for i,att in enumerate(attribs_m)}
    # Return the prediction dictionary and prediction time 
    pred_dict_list = []
    for i,att in enumerate(attribs_m):
      pred_dict = {pred_classes[i]:conf_classes[i]}
      pred_dict_list.append(pred_dict)
    #print(pred_dict_list)
    return pred_dict_list[0],pred_dict_list[1],pred_dict_list[2],pred_dict_list[3],pred_dict_list[4],pred_dict_list[5]
def generate_out_response(prompt, y_out, image_path, classifier,attribs=attribs_m):
    """Formats and prints the output of the combined model in a GenAI style, line by line"""
    original_text = prompt
    #prompt1 = input("prompts")
    candidate_labels = ['pool','condition', 'material', 'all', 'type','tree','roof','solar']
    out_dict ={attribs[at]:y_out[at] for at in range(len(attribs))}
    def clean_prompt(prompt):
      replace_items = ['.','building','image','roof?','roof.']
      prompt = prompt.lower()
      for r in replace_items:
        prompt = prompt.replace(r,'')
      #print(prompt)
      return prompt
    def att_selector(result:str):
      attribs_dict = {
          'condition':'Roof Condition',
          'material':'Roof Material',
          'type': 'Roof Style',
          'solar': 'Solar Panel',
          'tree': 'Tree Overhang',
          'pool': 'Swimming Pool'}
      if result == 'all':
        att_2_select = ['Roof Condition', 'Roof Material','Roof Style','Solar Panel', 'Tree Overhang', 'Swimming Pool']
      elif result == 'roof':
        att_2_select = ['Roof Condition', 'Roof Material','Roof Style']
      else:
        att_2_select = [attribs_dict[result]]
      return att_2_select
    out = classifier(clean_prompt(prompt), candidate_labels)
    result = out['labels'][0]
    att_2_select = att_selector(result)
    output_lines = [
        f"Image Analysis Results:\n",
        f"Image: {image_path}",
        f"User Input: \"{original_text}\"",
        f"Interpretation: ",
        #"It seems that the image strongly suggests the following:- "
        ]
    for key, value in out_dict.items():
      if key in att_2_select:
        output_lines.append(f"  * The {key} is {value}.")
    output_stream = ''
    for line in output_lines:
        output_stream += line+'\n'
    return output_stream

def multimodel_effnet_gr(image, text,model: torch.nn.Module, transform:torchvision.transforms,classifier: pipeline):
    # Replace this with your actual image and text processing logic
    #prompt1 = input("prompts")
    img = image.convert('RGB')
    img = transform(img).unsqueeze(0)
    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_logits = model(img)

    pred_label = pred_logits_to_labels(pred_logits)
    pred_classes = pred_label_2_classes(pred_label)
    #y_out = ['Fair','Shingle','Mixed','No','Low','No']
    result = generate_out_response(prompt=text, y_out=pred_classes,classifier = classifier, image_path="Image.jpg")
    return result

if __name__ == '__main__':
    #article = "Created https://github.com/omer770/multimodel_effnet.git"
  example_list  = [['/content/multimodel_effnet/data/Datasets2/Image_22884_Metal_Gabled_Good.jpg'],
                    ['/content/multimodel_effnet/data/Datasets2/Image_31333_Asphalt_Flat_Fair.jpg'],
                    ['/content/multimodel_effnet/data/Datasets2/Image_31078_Poly_Flat_Fair.jpg']]
  # Create the Gradio demo
  demo = gr.Interface(
      fn=predict,
      inputs=gr.Image(type="pil"),
      outputs=[
          gr.Label(label="Roof Condition"),
          gr.Label(label="Roof Material"),
          gr.Label(label="Roof Style"),
          gr.Label(label="Solar Panel"),
          gr.Label(label="Tree Overhang"),
          gr.Label(label="Swimming Pool"),
      ],
      examples=example_list, 
      title="<b>TWIA: Building Attribute Classifier</b>",
      description="An EfficientNetB2 model for analyzing roof condition, materials, and more.",
      article="View the project on GitHub",
      css="""
          .gradio-container h1 { font-size: 22px; } 
          .gradio-container a { color: #007bff; text-decoration: none; }
          .gradio-container a:hover { text-decoration: underline; }
      """,)

  # Launch the demo!
  demo.launch(debug=False, # print errors locally?
              share=True) # generate a publically shareable URL?

  iface = gr.Interface(
    fn=multimodel_effnet_gr,
    inputs=[gr.Image(type='pil'), "text"],  
    outputs="text",
    examples=example_list, 
    title="<b>TWIA: Building Attribute Classifier</b>",
    description="An EfficientNetB2 model for analyzing roof condition, materials, and more.",
    article="View the project on GitHub",
    css="""
        .gradio-container h1 { font-size: 22px; } 
        .gradio-container a { color: #007bff; text-decoration: none; }
        .gradio-container a:hover { text-decoration: underline; }
    """,)

  iface.launch(debug=True)
