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
NEW_LINE_CHAR = "\n"

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
class Get_and_process_data:
    def __init__(self, data_path = DEFAULT_DATA_DIR, train_only = True):
        
        print(data_path)
        
        self.labels = ["test", "treatment", 
                    "present", "absent", "possible", "conditional",
                    "hypothetical", "associated with someone else"]
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
        raw_text = raw_files[["text", "filename"]].set_index("filename")
        
        ast_concept_df = pd.DataFrame()
        for i, file in raw_files.iterrows():
            ast_dict = file["ast"]
            concept_dict = file["concept"]
            tmp_ast = pd.DataFrame(ast_dict)
            tmp_ast = tmp_ast.drop(columns=["concept_type"])
            tmp_ast = tmp_ast.rename(columns={"assertion_type": "ast_con_label"})

            #Only concepts with not "problem"
            tmp_concept = pd.DataFrame(concept_dict)
            if len(tmp_ast) > 0:
                assert(
                    tmp_concept[tmp_concept["concept_type"]=="problem"]["concept_text"].reset_index(drop=True).equals(
                        tmp_ast["concept_text"]
                    )
                ), "Concepts with problem type are not the same as assertions"
            tmp_concept = tmp_concept.rename(columns={"concept_type": "ast_con_label"})
            tmp = tmp_concept[tmp_concept["ast_con_label"]=="problem"].reset_index(drop=True)
            tmp["ast_con_label"] = tmp_ast["ast_con_label"]
            tmp_ast = tmp
            tmp_concept = tmp_concept[tmp_concept["ast_con_label"] != "problem"]
            
            tmp_ast["filename"] = file["filename"]
            tmp_concept["filename"] = file["filename"]
            if len(tmp_ast) > 0:
                ast_concept_df = ast_concept_df.append(tmp_ast, ignore_index=True)
            if len(tmp_concept) > 0:
                ast_concept_df = ast_concept_df.append(tmp_concept, ignore_index=True)            
        #cols = concept_text, start_line, start_word_number, end_line, end_word_number, ast_con_label, filename
        
        return ast_concept_df, raw_text

    def format(self, ast_concept_df : pd.DataFrame, raw_text : pd.DataFrame):
        
        preproc_data = {}

        for i, row in tqdm(ast_concept_df.iterrows()):
            filename = row["filename"]
            text = raw_text.loc[filename]["text"]

            # text Normalization
            text = text.lower()
            line = text.split(NEW_LINE_CHAR)[row["start_line"] - 1]  # NOTE: we assume that start_line == end_line
            line = " ".join(line.split()) # remove multiple spaces
            row["concept_text"] = " ".join(row["concept_text"].split()) # remove multiple spaces

            # find character index start and end of concept
            start_char_index = len(" ".join(line.split()[: row["start_word_number"]]))  # number of chars before concept
            if start_char_index > 0:
                start_char_index += 1
            end_char_index = start_char_index + len(row["concept_text"])
            if line[start_char_index:end_char_index] == "cut cord compression , par":
                print("dlmds")
            assert (
                line[start_char_index:end_char_index] == row["concept_text"]
            ), f"concept_text doesn't match the found indexes. '{line[start_char_index:end_char_index]}' != '{row['concept_text']}'"

            line_id = filename + "_" + str(row["start_line"])
            if line_id not in preproc_data:
                preproc_data[line_id] = {
                    "text": line,
                }
                for label in self.labels:
                    preproc_data[line_id][label] = []
                    # use sets because the indices can repeat for various reasons
                    preproc_data[line_id][label+"_indices_start"] = set()
                    preproc_data[line_id][label+"_indices_end"] = set()
            for label in self.labels:
                if row["ast_con_label"] == label:
                    preproc_data[line_id][label].append(row["concept_text"])
                    preproc_data[line_id][label+"_indices_start"].add(start_char_index)
                    preproc_data[line_id][label+"_indices_end"].add(end_char_index)
                    break
        return pd.DataFrame(preproc_data).T
        
        
if __name__ == "__main__":
    data = Get_and_process_data()
    print(data.format(*data.load_parse()))