from pickle import dump
from pickle import load


# save the model
def save_model(model):
    dump(model, open(str(model)+'.pkl', 'wb'))

# load the model
def load_model(model):
    my_model = load(open(str(model)+'.pkl', 'rb'))
    return my_model
