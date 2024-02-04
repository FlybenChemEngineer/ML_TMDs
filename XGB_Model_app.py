import xgboost as xgb
import pandas as pd
import joblib
import os
from Function_env import loaddata, data_process, Stand_data

def predict_with_xgb_model(features, model_path):
    # 加载已经训练好的模型
    trained_model = xgb.Booster()
    trained_model.load_model(model_path)

    # 将特征数据转换为 DMatrix 对象
    features_dmatrix = xgb.DMatrix(features)

    # 使用模型进行预测
    predictions = trained_model.predict(features_dmatrix)

    # 返回预测结果
    return predictions



folder_name = "Dataset"
result = loaddata.read_excel_files_in_folder(folder_name)

file_to_read = "dataset_240131_test.xlsx"

if file_to_read in result:
    data_to_read = result[file_to_read]

features, ulstress, moduls, feature_labels = data_process.features_targets_read(data_to_read)

user_home = os.path.expanduser("~")

save_folder = os.path.join(user_home, 'machine_learning_space', 'DM', 'Performance', 'scaler_0131.pkl')
model_save_folder = os.path.join(user_home, 'machine_learning_space', 'DM', 'Performance')
model_path = os.path.join(model_save_folder, 'XGB_moduls_model.model')

scaler = joblib.load(save_folder)

features = scaler.transform(features)

pred = predict_with_xgb_model(features, model_path)

pred = pd.DataFrame(pred)
pred.to_excel(os.path.join(model_save_folder, 'moduls_pred.xlsx'), index=False)

print(pred)