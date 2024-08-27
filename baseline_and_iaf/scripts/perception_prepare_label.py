import os, sys
import pandas as pd
import numpy as np
import pathlib

import config
from data_parser import load_data
from config_model_feat import label_dims

# prepare df_label
task='perception'
feature='egemaps'
paths={'features': config.PATH_TO_FEATURES[task],
    'labels': config.PATH_TO_LABELS[task],
    'partition': os.path.join(config.PATH_TO_METADATA[task], f'partition.csv')
    }

appended_data = []
for _label in label_dims[:]:
    paths.update({'data': os.path.join(config.DATA_FOLDER, task, _label)})
    data = load_data(task, paths, feature, _label)
    labels=np.array(data['devel']['label']).flatten()
    meta=np.array(data['devel']['meta']).flatten()
    _df = pd.DataFrame(data=labels,  
                    index=meta,  
                    columns=[_label])
    _df.index.names=['subj_id']
    appended_data.append(_df)
df_label=pd.concat(appended_data, axis=1)
#print(df_label.shape, df_label.head(3))

# write output
csv_path=os.path.join(f'results/prediction_muse/perception/lf/label_devel.csv')
csv_dir = pathlib.Path(csv_path).parent.resolve()
os.makedirs(csv_dir, exist_ok=True)

df_label.to_csv(csv_path)
print(f'Prediction file written to {csv_path}.')
