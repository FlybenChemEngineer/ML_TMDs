from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import pandas as pd
import os

def standardize_data(data, save = False):

    scaler = StandardScaler()

    standardized_data = scaler.fit_transform(data)

    #standardize_data = pd.DataFrame(standardize_data)

    # 获取用户主目录
    user_home = os.path.expanduser("~")

    # 定义保存路径
    save_folder = os.path.join(user_home, 'machine_learning_space', 'DM', 'Performance')

    #save_folder = 'Performance'


    if save == True:
        #joblib.dump(scaler, 'scaler_0131_ulstress.pkl')

        scaler_path = os.path.join(save_folder, 'scaler_0131.pkl')
        joblib.dump(scaler, scaler_path)

        print('scaler.pkl is saved')

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print('Data normalization has been completed                     ', current_time)


    return standardized_data