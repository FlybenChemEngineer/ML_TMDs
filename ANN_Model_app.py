from Function_env import loaddata, data_process, Model_ensemble, Stand_data
import joblib
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score


folder_name = "Dataset"
result = loaddata.read_excel_files_in_folder(folder_name)

file_to_read = "dataset_240131_test.xlsx"
#file_to_read = "dataset_240128.xlsx"


if file_to_read in result:
    data_to_read = result[file_to_read]

features, ulstress, moduls, feature_labels = data_process.features_targets_read(data_to_read)


user_home = os.path.expanduser("~")

save_folder = os.path.join(user_home, 'machine_learning_space', 'DM', 'Performance', 'scaler_0131.pkl')

scaler = joblib.load(save_folder)

features = scaler.transform(features)

#print(features)

#print(ulstress)

def preprocess_data(features, targets):
    def process_input(input_data):
      
        if isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 1:
    
                return torch.tensor(input_data.reshape(-1, 1), dtype=torch.float32)
            else:
                return torch.tensor(input_data, dtype=torch.float32)
        elif isinstance(input_data, pd.DataFrame):
            data_as_numpy = input_data.to_numpy()
            if len(data_as_numpy.shape) == 1:

                return torch.tensor(data_as_numpy.reshape(-1, 1), dtype=torch.float32)
            else:
                return torch.tensor(data_as_numpy, dtype=torch.float32)
        else:
            raise ValueError("Unsupported data type. Please provide NumPy array or DataFrame.")
        
    processed_features = process_input(features)
    processed_targets = process_input(targets)

    return processed_features, processed_targets

class ComplexANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ComplexANN, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

features, targets = preprocess_data(pd.DataFrame(features),pd.DataFrame(ulstress))   

input_size = features.shape[1]
output_size = targets.shape[1]
hidden_sizes = [256, 128, 64]

model_folder = os.path.join(user_home, 'machine_learning_space', 'DM', 'Performance', 'ANN_Ulstress_model.pth')
model = ComplexANN(input_size, hidden_sizes, output_size)

model.load_state_dict(torch.load(model_folder))

model.eval()

with torch.no_grad():
    y_case_pred = model(features).detach().numpy()

# 定义保存路径
save_case_folder = os.path.join(user_home, 'machine_learning_space', 'DM', 'Performance')

# 确保文件夹存在
os.makedirs(save_case_folder, exist_ok=True)
y_case_pred_df = pd.DataFrame(y_case_pred)

y_case_pred_df.to_excel(os.path.join(save_case_folder, 'y_case_pred.xlsx'), index=False)


print(y_case_pred)




#print(features)
