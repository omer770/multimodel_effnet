import torch

class effnet_multimodel(torch.nn.Module):
    # Initialize the parameters
    def __init__(self, model_name,effnet, weights):
        super(effnet_multimodel, self).__init__()
        self.weights = weights
        self.model_name = model_name
        self.effnet = effnet(weights=self.weights)
        self.effnet.classifier = torch.nn.Identity()
        self.drp1 = torch.nn.Dropout(0.2)
        self.f1 = torch.nn.Flatten()
        self.dense3 = torch.nn.ReLU(torch.nn.Linear(64,128))
        self.drp3 = torch.nn.Dropout(0.5)
        self.dense4 =  torch.nn.ReLU(torch.nn.Linear(128,32))
        self.drp4 = torch.nn.Dropout(0.5)
        self.output_condition =  torch.nn.Linear(1280,4)
        self.output_material = torch.nn.Linear(1280,6)
        self.output_type = torch.nn.Linear(1280,4)
        self.output_tree = torch.nn.Linear(1280,4)
        self.output_pool = torch.nn.Linear(1280,3)
        self.output_solar = torch.nn.Linear(1280,2)

    def forward(self,inputs):
        x = self.effnet(inputs)
        x = self.drp1(x)
        x = self.f1(x)
        x = self.dense3(x)
        x = self.drp3(x)
        x = self.dense4(x)
        x = self.drp4(x)
        out_c = self.output_condition(x)
        out_m = self.output_material(x)
        out_s = self.output_solar(x)
        out_y = self.output_type(x)
        out_t = self.output_tree(x)
        out_p = self.output_pool(x)
        #out_Variable = tensor = torch.Tensor([out_c,out_m,out_y,out_s,out_t,out_p])
        return out_c,out_m,out_y,out_s,out_t,out_p