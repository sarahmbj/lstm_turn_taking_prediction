import torch
from pprint import pprint


def load_model(pickled_model):
    with open(pickled_model, "rb") as model_file:
        if torch.cuda.is_available():
            model = torch.load(model_file)
        else:
            model = torch.load(model_file, map_location=torch.device('cpu'))
        print(type(model))
        pprint(model)


load_model('results/example_pickled_model/model.p')
