import os
import pandas as pd

def read_excel_files_in_folder(folder_path):

    current_path = os.path.dirname(os.path.abspath(__file__))
    

    folder_full_path = os.path.join(current_path, folder_path)
    

    files = os.listdir(folder_full_path)
    
    all_data = {}

    for file in files:
     
        if file.endswith(".xlsx") or file.endswith(".xls"):
        
            file_full_path = os.path.join(folder_full_path, file)
            
          
            excel_data = pd.read_excel(file_full_path)
            
            all_data[file] = excel_data

    return all_data

if __name__ == "__main__":
  
    folder_name = "Dataset"
    result = read_excel_files_in_folder(folder_name)


    for file_name, data in result.items():
        print(f"Data from {file_name}:")
        print(data)
        print("\n")