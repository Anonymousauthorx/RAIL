from marvin.models.marvin import MARVIN, NoLSTM
from marvin.models.other import GVIN, GATNet

models = {
            'gvin': GVIN,
            'nolstm': NoLSTM,
            'gat': GATNet,
            'vin': MARVIN,
}
