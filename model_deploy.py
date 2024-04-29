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
suggestions_dict = {
    "Roof Condition": {
        "Fair": "Your roof is in fair condition. While the roof currently functional, consider these actions to prolong its lifespan.Schedule a professional inspection, perform proactive maintenance, and plan for future replacement",
        "Good": "The roof is in good condition. Regular inspections and maintenance will help ensure its longevity",
        "Poor": "The roof shows signs of patch or wear and tear or rust. Consider a professional roof inspection to determine if repairs or replacement are necessary.The roof requires an immediate attention",
        "Damaged": "The roof has sustained damage. A thorough inspection and likely repairs are needed to prevent further deterioration"
    },
    "Roof Material": {
        "Metal": "Metal roofs are known for their durability and longevity. Here are some things to keep in mind when having metal roofs: regular maintenance, potential sound dampening measures, and suitability for your specific climate.Metal roof is a good choice for barn but if it is for Residential stucture,switching to Shingle or Tile is a better option",
        "Poly": "Poly roofs are a less common option.These are common for Commercial buildings. For Poly roofs, its important to research their durability, maintenance needs, and suitability for your climate",
        "Shingle": "Shingle roofs are a popular choice, offering a balance of affordability and durability. Make sure to choose shingles appropriate for your area’s weather conditions",
        "Tile": "Tile roofs are known for their longevity and beauty but can be more expensive. Tile roofs are great chioce for warmer climates", 
        "Ballasted": "Ballasted roofs are typically used on flat commercial roofs. Ballasted roof consist of layers of gravel and asphalt for protection",
        "Asphalt": "Asphalt is often used in roll roofing or as a base material for shingles. Asphalt is affordable but less durable than other options"
    },
    "Roof Style": {
        "Flat": "Flat roofs are more common on commercial buildings. Ensure proper drainage and regular inspections to prevent water ponding issues",
        "Gabled": "A classic and simple roof design with two sloping sides. Gabled roofs offer good water shedding capabilities",
        "Hip": "Hip roofs feature slopes on all four sides. They provide good wind resistance and stability",
        "Mixed": "Mixed roof styles combine elements of different styles. Mixed roof can be more complex, so it’s important to work with an experienced roofer" 
    },
    "Solar Panel": {
        "No": "If you are considering solar panels in the future, Give your installation requirements.We offer SOTA solar design installations.Using AI we predict which facets have more exposer to sunlight and provide optimal layout for installing solar panals",
        "Solar_Panel-Yes": "Ensure your roof is in good structural condition to support the weight of solar panels"
      },
    "Tree Overhang": {
        "Low": "Minor tree overhang may not be a significant concern. Monitor the roof for accumulating debris",
        "Medium": "Medium tree overhang can shade the roof and lead to moisture issues. Consider trimming branches",
        "High": "Significant tree overhang poses a risk of falling branches and debris. Have the trees professionally evaluated and trimmed as needed", 
        "No": "No tree overhang means better sunlight exposure and potentially less debris accumulation"
      },
    "Swimming Pool": {
        "No": "No swimming pool present means less concern about the roof being exposed to excess moisture",
        "Yes": "A swimming pool near the structure can increase humidity levels. Be observant of potential moisture-related issues on the roof",
        "Screened": "A screened pool reduces the amount of direct water and debris exposure to the roof.Its a good choice having a Screened pool"
    }
}

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def pred_label_2_classes (pred_label,label_2_item_lst=label_2_item_lst):
  pred_class = [label_2_item_lst[i][label] for i,label in enumerate(pred_label)]
  return pred_class

def pred_logits_to_labels(pred_logits,get_conf:bool = False):
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
  if get_conf:
    pred_conf = [round(conf_c.max().cpu().item(),2),round(conf_m.max().cpu().item(),2),
                        round(conf_y.max().cpu().item(),2),round(conf_s.max().cpu().item(),2),
                        round(conf_t.max().cpu().item(),2),round(conf_p.max().cpu().item(),2)]
    return pred_label , pred_conf
  else: return pred_label


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
            pred_logits = model(transformed_image) # perform inference on target sample 
            pred_label = pred_logits_to_labels(pred_logits)
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

def predict(img,model, transform) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    if transform:
      # Transform the target image and add a batch dimension
      img = transform(img)
    img = img.unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_logits = model(img)
    pred_label,pred_conf = pred_logits_to_labels(pred_logits,get_conf = True)
    pred_classes = pred_label_2_classes(pred_label)
    conf_classes = pred_conf
    # Calculate the prediction time
    #pred_classes_dict = {att:(pred_classes[i],conf_classes[i]) for i,att in enumerate(attribs_m)}
    # Return the prediction dictionary and prediction time 
    pred_dict_list = []
    for i,att in enumerate(attribs_m):
      pred_dict = {pred_classes[i]:conf_classes[i]}
      pred_dict_list.append(pred_dict)
    #print(pred_dict_list)
    return pred_dict_list[0],pred_dict_list[1],pred_dict_list[2],pred_dict_list[3],pred_dict_list[4],pred_dict_list[5]


def generate_out_response(prompt, y_out, classifier,suggestions_dict=suggestions_dict,attribs=attribs_m):
    """Formats and prints the output of the combined model in a GenAI style, line by line"""
    original_text = prompt
    #prompt1 = input("prompts")
    candidate_labels = ['pool','condition', 'material', 'all', 'type','tree','roof','solar']
    out_dict ={attribs[at]:y_out[at] for at in range(len(attribs))}
    def clean_prompt(prompt):
      replace_items = ['building','image','roof?','roof.','buliding.','.','property','analyze']
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
        f"User Input: \"{original_text}\"",
        f"Interpretation: ",
        #"It seems that the image strongly suggests the following:- "
        ]
    for key, value in out_dict.items():
      if key in att_2_select:
        output_lines.append(f"  * The {key} is {value}.")
    output_lines.append(f"\nSuggestions: ")
    for key, value in out_dict.items():
      if key in att_2_select:
        suggestions_lists = suggestions_dict[key][value].split(".")
        #suggestions_lists.remove('.')
        for suggestions in suggestions_lists:
          output_lines.append(f"  * {suggestions.strip()}.")
    output_stream = ''
    for line in output_lines:
        output_stream += line+'\n'    
        
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
