import pickle


def test_on_model(pickled_model):
    with open(pickled_model, "rb") as model_file:
        model = pickle.load(model_file)
        print(type(model))
        print(model)



test_on_model('results/example_pickled_model/model.p')