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

    def load_parse(self):
        """See EDA for better understanding of this function"""
        
        text_files = glob.glob(self.train_data_path + os.sep + self.txt_folder_name + os.sep +  "*.txt")
        filename = ""
        raw_files = pd.DataFrame()
        for file in tqdm(text_files):
            with open(file, 'r') as f:
                text = f.read()
                filename = file.split("/")[-1].split(".")[0]
                ast = parse_ast(self.train_data_path + os.sep + self.ast_folder_name + os.sep +  filename + ".ast")
                concept = parse_concept(self.train_data_path + os.sep + self.concept_folder_name + os.sep +  filename + ".con")
                rel = parse_relation(self.train_data_path + os.sep + self.rel_folder_name + os.sep +  filename + ".rel")
                
                raw_files = raw_files.append(pd.DataFrame({"text": [text], "filename": [filename] , "concept": [concept], "ast": [ast], "rel": [rel]}), ignore_index=True)
        
        
        ast_concept_df = pd.DataFrame()
        for i, file in raw_files.iterrows():
            ast_dict = file["ast"]
            concept_dict = file["concept"]
            tmp_ast = pd.DataFrame(ast_dict)
            tmp_ast = tmp_ast.drop(columns=["concept_type"])
            tmp_ast = tmp_ast.rename(columns={"assertion_type": "ast_con_label"})

            #Only concepts with not "problem"
            tmp_concept = pd.DataFrame(concept_dict)
            tmp_concept = tmp_concept[tmp_concept["concept_type"] != "problem"]
            tmp_concept = tmp_concept.rename(columns={"concept_type": "ast_con_label"})
            
            tmp_ast["filename"] = file["filename"]
            tmp_concept["filename"] = file["filename"]
            ast_concept_df = ast_concept_df.append(tmp_ast, ignore_index=True)
            ast_concept_df = ast_concept_df.append(tmp_concept, ignore_index=True)
        #cols = concept_text, start_line, start_word_number, end_line, end_word_number, ast_con_label, filename
        
        return ast_concept_df

    # def get_dataset(self):
        
        
        
if __name__ == "__main__":
    print(Get_and_process_data().load_parse())