# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
from datasets import Dataset, load_dataset, Sequence, ClassLabel
from tqdm import tqdm
from transformers import (AutoTokenizer)
import pandas as pd
import glob
import sys
import os

def get_root_dir():
    path = os.path.dirname(__file__)
    for _ in range(1):
        path = os.path.dirname(path)
    return path



sys.path.append(get_root_dir())
from src.utils.parse_data import parse_ast, parse_concept, parse_relation

# ---------------------------------------------------------------------------- #
#                                   constants                                  #
# ---------------------------------------------------------------------------- #
DEFAULT_DATA_DIR = os.path.join(get_root_dir(), "data")
NEW_LINE_CHAR = "\n"
MODEL_CHECKPOINT =  "allenai/scibert_scivocab_uncased"

# ---------------------------------------------------------------------------- #
#                                     Utils                                    #
# ---------------------------------------------------------------------------- #
def get_simple_tokenize(tokenizer):
    def simple_tokenize(row):
        tokens = tokenizer(row["text"], return_offsets_mapping=True)    
        return tokens
    return simple_tokenize

def get_generate_row_labels(tokenizer, label_list, available_labels, verbose=False):
    def generate_row_labels(row, verbose=verbose):
        """ Given a row from the consolidated `Ade_corpus_v2_drug_ade_relation` dataset, 
        generates BIO tags for drug and effect entities. 
        
        """

        text = row["text"]

        labels = []
        label = "O"
        prefix = ""
        
        # while iterating through tokens, increment to traverse all drug and effect spans
        label_index = {l : 0 for l in available_labels}
        
        tokens = tokenizer(text, return_offsets_mapping=True)

        for n in range(len(tokens["input_ids"])):
            offset_start, offset_end = tokens["offset_mapping"][n]

            # should only happen for [CLS] and [SEP]
            if offset_end - offset_start == 0:
                labels.append(-100)
                continue
            
            
            for l in available_labels :
                if label_index[l] < len(row[l+"_indices_start"]) and offset_start == row[l+"_indices_start"][label_index[l]]:
                    label = l.upper()
                    prefix = "B-"
                    break
            
            labels.append(label_list.index(f"{prefix}{label}"))
            
            for l in available_labels :
                if label_index[l] < len(row[l+"_indices_start"]) and offset_end == row[l+"_indices_start"][label_index[l]]:
                    label = "O"
                    prefix = ""
                    label_index += 1
                    break

            # need to transition "inside" if we just entered an entity
            if prefix == "B-":
                prefix = "I-"
        
        if verbose:
            print(row)
            orig = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
            for n in range(len(labels)):
                print(orig[n], labels[n])
        tokens["labels"] = labels
        
        return tokens
    return generate_row_labels

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
class Get_and_process_data:
    def __init__(self, tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT), 
                 data_path = DEFAULT_DATA_DIR, train_split = 0.8, add_unlabeled=True):
        self.labels = ["test", "treatment", 
                    "present", "absent", "possible", "conditional",
                    "hypothetical", "associated with someone else"]
        self.train_data_path = os.path.join(data_path, "train")
        self.test_data_path = os.path.join(data_path, "val")
        self.ast_folder_name = "ast"
        self.concept_folder_name = "concept"
        self.rel_folder_name = "rel"
        self.txt_folder_name = "txt"
        self.train_split = train_split
        self.tokenizer = tokenizer
        self.label_list = None
        self.add_unlabeled = add_unlabeled
    
    def load_raw_test(self):
        text_files = glob.glob(self.test_data_path + os.sep + self.txt_folder_name + os.sep +  "*.txt")
        filename = ""
        lines = pd.DataFrame()
        for file in tqdm(text_files, "Loading raw text for test"):
            with open(file, 'r') as f:
                text = f.read()
                filename = file.split("/")[-1].split(".")[0]
                for i, line in enumerate(text.split(NEW_LINE_CHAR)): 
                    lines = lines.append(pd.DataFrame({"text": [line], "filename": [filename], "line": i+1}), ignore_index=True)

        preproc_data = {}
        for i, row in tqdm(lines.iterrows(), "Formatting test data"):
            line_id = row["filename"] + "_" + str(row["line"])
            preproc_data[line_id] = {
                    "text": row["text"],
                }
        
        # -------------------- Save and make hugging face dataset -------------------- #
        preproc_df = pd.DataFrame(preproc_data).T
        preproc_df.to_json(os.path.join(self.test_data_path, "dataset.jsonl"), orient="records", lines=True)
        dataset = load_dataset("json", data_files={"test" : os.path.join(self.test_data_path, "dataset.jsonl")})
        
        # ------------------ Creating the right format and tokenize ------------------ #
        label_list = ['O']
        for label in self.labels:
            label_list.append("B-"+label.upper())
            label_list.append("I-"+label.upper())
            
        custom_seq = Sequence(feature=ClassLabel(num_classes=len(label_list),
                                                names=label_list,
                                                names_file=None, id=None), length=-1, id=None)

        dataset["test"].features["ner_tags"] = custom_seq
        
        labeled_dataset = dataset.map(get_simple_tokenize(self.tokenizer))
        return labeled_dataset
        
    def load_parse(self):
        """
        Output :
        
        ast_concept_df :
                    concept_text  start_line  start_word_number  end_line  end_word_number ast_con_label      filename
        0                     pain          55                 10        55               10  hypothetical  018636330_DH.txt
        1           hyperlipidemia          29                  4        29                4       present  018636330_DH
        
        raw_text :
                                                                    text
        filename                                                          
        018636330_DH     018636330 DH\n5425710\n123524\n0144918\n6/2/20...
        """
        text_files = glob.glob(self.train_data_path + os.sep + self.txt_folder_name + os.sep +  "*.txt")
        filename = ""
        raw_files = pd.DataFrame()
        unlabeled_lines = pd.DataFrame()
        for file in tqdm(text_files, "Loading raw text"):
            with open(file, 'r') as f:
                text = f.read()
                filename = file.split("/")[-1].split(".")[0]
                ast = parse_ast(self.train_data_path + os.sep + self.ast_folder_name + os.sep +  filename + ".ast")
                concept = parse_concept(self.train_data_path + os.sep + self.concept_folder_name + os.sep +  filename + ".con")
                rel = parse_relation(self.train_data_path + os.sep + self.rel_folder_name + os.sep +  filename + ".rel")
                
                raw_files = raw_files.append(pd.DataFrame({"text": [text], "filename": [filename] , "concept": [concept], "ast": [ast], "rel": [rel]}), ignore_index=True)
            # -------------------- known lines are the ones in Concept ------------------- #
            known_lines = concept["start_line"]
            for i, line in enumerate(text.split(NEW_LINE_CHAR)): 
                if not i+1 in known_lines:
                    unlabeled_lines = unlabeled_lines.append(pd.DataFrame({"text": [line], "filename": [filename], "line": i+1}), ignore_index=True)
        raw_text = raw_files[["text", "filename"]].set_index("filename")
        
        # ---------------------------- Mixing the 2 tasks ---------------------------- #
        ast_concept_df = pd.DataFrame()
        for i, file in tqdm(raw_files.iterrows(), "Processing raw text"):
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
        
        return ast_concept_df, raw_text, unlabeled_lines

    def format(self, ast_concept_df : pd.DataFrame, raw_text : pd.DataFrame, unlabeled_lines : pd.DataFrame):
        
        preproc_data = {}

        # -------------------------- Add the unlabeled_lines ------------------------- #
        for i, row in tqdm(unlabeled_lines.iterrows(), "Adding unlabeled lines"):
            line_id = row["filename"] + "_" + str(row["line"])
            preproc_data[line_id] = {
                    "text": row["text"],
                }
            for label in self.labels:
                preproc_data[line_id][label] = []
                # use sets because the indices can repeat for various reasons
                preproc_data[line_id][label+"_indices_start"] = set()
                preproc_data[line_id][label+"_indices_end"] = set()
        
        for i, row in tqdm(ast_concept_df.iterrows(), "Formatting dataset"):
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
        preproc_df = pd.DataFrame(preproc_data).T
        for label in self.labels:
            preproc_df[label+"_indices_start"] = preproc_df[label+"_indices_start"].apply(list).apply(sorted)
            preproc_df[label+"_indices_end"] = preproc_df[label+"_indices_end"].apply(list).apply(sorted)

        preproc_df.to_json(os.path.join(self.train_data_path, "dataset.jsonl"), orient="records", lines=True)
        dataset = load_dataset("json", data_files=os.path.join(self.train_data_path, "dataset.jsonl"))
        dataset = dataset["train"].train_test_split(train_size=self.train_split)
        # ---------------------------------- rename ---------------------------------- #
        dataset["val"] = dataset["test"]
        return dataset
    
    def token_labeling(self, dataset : Dataset):
        
        label_list = ['O']
        for label in self.labels:
            label_list.append("B-"+label.upper())
            label_list.append("I-"+label.upper())
            
        custom_seq = Sequence(feature=ClassLabel(num_classes=len(label_list),
                                                names=label_list,
                                                names_file=None, id=None), length=-1, id=None)

        dataset["train"].features["ner_tags"] = custom_seq
        dataset["val"].features["ner_tags"] = custom_seq
        
        labeled_dataset = dataset.map(get_generate_row_labels(self.tokenizer, label_list, self.labels))
        self.label_list = label_list
        
        # ----------------------------- Adding test data ----------------------------- #
        test_dataset = self.load_raw_test()
        labeled_dataset["test"] = test_dataset["test"]
        return labeled_dataset

    def get_label_list(self):
        return self.label_list
    
    def get_dataset(self):
        dataset = self.format(*self.load_parse())
        return self.token_labeling(dataset)
def get_default_dataset():
    return Get_and_process_data().get_dataset()

if __name__ == "__main__":
    print(get_default_dataset())