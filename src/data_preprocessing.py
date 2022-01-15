# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
from asyncio.proactor_events import constants
from tqdm import tqdm
import pandas as pd
import glob
import sys
import os

def get_root_dir():
    path = os.path.dirname(__file__)
    while "src" in path:
        path = os.path.dirname(path)
    return path



sys.path.append(get_root_dir())
from src.utils.parse_data import parse_ast, parse_concept, parse_relation

# ---------------------------------------------------------------------------- #
#                                   constants                                  #
# ---------------------------------------------------------------------------- #
DEFAULT_DATA_DIR = os.path.join(get_root_dir(), "data")


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
class Get_and_process_data:
    def __init__(self, data_path = DEFAULT_DATA_DIR):
        
        print(data_path)
        
        self.train_data_path = os.path.join(data_path, "train")
        self.val_data_path = os.path.join(data_path, "val")
        self.ast_folder_name = "ast"
        self.concept_folder_name = "concept"
        self.rel_folder_name = "rel"
        self.txt_folder_name = "txt"

    def preprocess(self):
        text_files = glob.glob(self.train_data_path + os.sep + self.txt_folder_name + os.sep +  "*.txt")
        filename = ""
        df = pd.DataFrame()
        for file in tqdm(text_files):
            with open(file, 'r') as f:
                text = f.read()
                filename = file.split("/")[-1].split(".")[0]
                ast = parse_ast(self.train_data_path + os.sep + self.ast_folder_name + os.sep +  filename + ".ast")
                concept = parse_concept(self.train_data_path + os.sep + self.concept_folder_name + os.sep +  filename + ".con")
                rel = parse_relation(self.train_data_path + os.sep + self.rel_folder_name + os.sep +  filename + ".rel")
                
                df = df.append(pd.DataFrame({"text": [text], "filename": [filename] , "concept": [concept], "ast": [ast], "rel": [rel]}), ignore_index=True)
        
        
        return df

    # def get_dataset(self):
        
        
        
if __name__ == "__main__":
    print(Get_and_process_data().preprocess())