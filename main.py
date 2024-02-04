from Function_env import loaddata, data_process, Model_ensemble, Stand_data

folder_name = "Dataset"
result = loaddata.read_excel_files_in_folder(folder_name)

file_to_read = "dataset_240128.xlsx"

if file_to_read in result:
    data_to_read = result[file_to_read]

features, ulstress, moduls, feature_labels = data_process.features_targets_read(data_to_read)

features = Stand_data.standardize_data(features, save = False)

print(features)

#result = Model_ensemble.modelcal(features, ulstress)
#result = Model_ensemble.modelcal(features, moduls)

