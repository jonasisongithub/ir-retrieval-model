# load data

from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
import pyterrier as pt

from load_dataset import load_dataset 
from create_index import create_index
from create_model import create_model
from test_model import test_model

if not pt.started():
    pt.init()

ensure_pyterrier_is_loaded()

# load data
documents = load_dataset()

# create index
index = create_index(documents)

# create model
model = create_model(index)

# test model
test_model(model)

