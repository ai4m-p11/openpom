import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from openpom.models.mpnn_pom import MPNNPOMModel
import torch
import numpy as np

TASKS = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

print("No of tasks: ", len(TASKS))
n_tasks = len(TASKS)

odour_model = MPNNPOMModel(n_tasks = n_tasks,
                        batch_size=128,
                        learning_rate=0.001, #learning_rate,
                        class_imbalance_ratio = list(np.random.rand(n_tasks)), #train_ratios,
                        loss_aggr_type = 'sum',
                        node_out_feats = 100,
                        edge_hidden_feats = 75,
                        edge_out_feats = 100,
                        num_step_message_passing = 5,
                        mpnn_residual = True,
                        message_aggregator_type = 'sum',
                        mode = 'classification',
                        number_atom_features = GraphConvConstants.ATOM_FDIM,
                        number_bond_features = GraphConvConstants.BOND_FDIM,
                        n_classes = 1,
                        readout_type = 'set2set',
                        num_step_set2set = 3,
                        num_layer_set2set = 2,
                        ffn_hidden_list= [392, 392],
                        ffn_embeddings = 256,
                        ffn_activation = 'relu',
                        ffn_dropout_p = 0.12,
                        ffn_dropout_at_input_no_act = False,
                        weight_decay = 1e-5,
                        self_loop = False,
                        optimizer_name = 'adam',
                        log_frequency = 32,
                        model_dir = f'./ensemble_models/experiments_9',
                        device_name='cuda')
odour_model.restore(f"./ensemble_models/experiments_9/checkpoint2.pt")

def predict_odours(molecule):
    featurizer = GraphFeaturizer()
    x = featurizer.featurize([molecule])
    molecule_data = dc.data.NumpyDataset(X=x, ids=[molecule])
    preds = odour_model.predict(molecule_data)
    #print(preds[0])

    odours = np.argwhere(preds[0] > 0.5)
    #print(odours)
    return [TASKS[s] for s in odours.flatten()]

molecule = 'C=CC(C)(O)CCC=C(C)C'
odours = predict_odours(molecule)
print(odours)