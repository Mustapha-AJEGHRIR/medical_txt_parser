import pandas as pd
import numpy as np
from tqdm import tqdm

import transformers
from datasets import Dataset, ClassLabel, Sequence, load_dataset, load_metric
from spacy import displacy
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          DataCollatorForTokenClassification,
                          pipeline,
                          TrainingArguments, 
                          Trainer,
                          AutoConfig,
                        AutoModelForSequenceClassification,
                        AutoTokenizer,
                        DataCollatorWithPadding,
                        EvalPrediction,
                        Trainer,
                        TrainingArguments,
                        default_data_collator,
                        set_seed,)

assert transformers.__version__ >= "4.11.0"

# from src.utils.parse_data import parse_ast, parse_concept, parse_relation


# ---------------------------------------------------------------------------- #
#                              CONCEPTS DETECTIONS                             #
# ---------------------------------------------------------------------------- #

label_names = ["O", "B-PROBLEM", "I-PROBLEM", "B-TEST", "I-TEST", "B-TREATMENT", "I-TREATMENT"]
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model_folder_name = "debru 3la path dyal modèle w7ettuh hna"
model_checkpoint = f"models/{model_folder_name}"

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
effect_ner_model = pipeline(task="ner", model=model, tokenizer=tokenizer)

def detect_concept(raw_text):
    outputs = effect_ner_model(raw_text, aggregation_strategy ="simple")
    entities = []

    params = [{"text": sentence, "ents": entities, "title": None}]

    html = displacy.render(
        params,
        style="ent",
        manual=True,
        # jupyter=True,
        options={
            "colors": {
                "PROBLEM": "#f08080",
                "TEST": "#9bddff",
                "TREATMENT": "#ffdab9",
            },
        },
    )

    return outputs


# ---------------------------------------------------------------------------- #
#                           ASSERTIONS CLASSIFICATION                          #
# ---------------------------------------------------------------------------- #

label_list = ['present',
 'possible',
 'absent',
 'conditional',
 'hypothetical',
 'associated_with_someone_else']

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

model_name_or_path = "..."

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    # cache_dir=cache_dir,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    # cache_dir=cache_dir,
    label2id=label2id,
    id2label=id2label
)

def detect_assertions(raw_text):
    lines = raw_text.split('\n')
    df = pd.DataFrame({"text": lines, "line_number": range(len(lines))})
    
    concept_df = pd.DataFrame(detect_concept(raw_text))

    if concept_type == "problem":
        text = df[(df["filename"] == fname) & (df["line_number"] == start_line-1)].text.values[0]
        concept_df.append({"concept_text": concept_text, "text": text, "line_number":start_line})
        
    concept_df = pd.DataFrame(concept_df)
    df = concept_df[["line_number", "text", "concept_text"]]
    df.rename(columns={"text":"sentence1", "concept_text":"sentence2"}, inplace=True)

    predict_dataset = Dataset.from_pandas(df, preserve_index=False)
    
    predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    desc="Running tokenizer on prediction dataset",
                )
    
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
    predictions = np.argmax(predictions, axis=1)

    df["prediction"] = [label2ast[label] for label in predictions]

    return df



# ---------------------------------------------------------------------------- #
#                             RELATIONS EXTRACTION                             #
# ---------------------------------------------------------------------------- #
model_folder_name = "......."
model_checkpoint = f"models/{model_folder_name}"

def extract_relations(raw_text):
    # split lines
    lines = raw_text.split('\n')
    df = pd.DataFrame({"text": lines, "line_number": range(len(lines))})

    # add concepts  
    concepts = detect_concept(raw_text)
    rel_df = pd.DataFrame()
    
    concept_df = pd.DataFrame(concepts)
    test_concept_df = concept_df[concept_df["concept_type"] == "test"]
    problem_concept_df = concept_df[concept_df["concept_type"] == "problem"]
    treatment_concept_df = concept_df[concept_df["concept_type"] == "treatment"]

    # class test --> problem
    test_problem_df = pd.merge(test_concept_df, problem_concept_df, how="inner", on="start_line")

    # class treatment --> problem
    treatment_problem_df = pd.merge(treatment_concept_df, problem_concept_df, how="inner", on="start_line")

    # class problem --> problem
    problem_problem_df = pd.merge(problem_concept_df, problem_concept_df, how="inner", on="start_line")
    problem_problem_df = problem_problem_df[problem_problem_df["concept_text_x"] != problem_problem_df["concept_text_y"]] # TODO: remove duplicates ?

    rel_df = pd.concat([test_problem_df, treatment_problem_df, problem_problem_df], axis=0)
   
        
    rel_df = rel_df.sort_values(by=["filename", "start_line"])
    rel_df = rel_df.reset_index(drop=True)

    def preprocess_text(row):
        line =  df[(df["filename"] == row["filename"]) & (df["line_number"] == row["start_line"]-1)]["text"].values[0]
        # line = line.lower()
        line = " ".join(line.split()) # remove multiple spaces

        concept_text_x = "<< "+ " ".join(line.split()[row["start_word_number_x"]:row["end_word_number_x"]+1]) + " >>"
        concept_text_y = "[[ " + " ".join(line.split()[row["start_word_number_y"]:row["end_word_number_y"]+1]) + " ]]"
        start_word_number_x = row["start_word_number_x"]
        end_word_number_x = row["end_word_number_x"]
        start_word_number_y = row["start_word_number_y"]
        end_word_number_y = row["end_word_number_y"]

        if row["start_word_number_x"] > row["start_word_number_y"]:
            concept_text_x, concept_text_y = concept_text_y, concept_text_x
            start_word_number_x, start_word_number_y = start_word_number_y, start_word_number_x
            end_word_number_x, end_word_number_y = end_word_number_y, end_word_number_x
        text = " ".join(line.split()[: start_word_number_x] + [concept_text_x] + line.split()[end_word_number_x+1: start_word_number_y] + [concept_text_y] + line.split()[end_word_number_y+1:])

        row["text"] = text
        return row

    predict_df = rel_df.apply(preprocess_text, axis=1)
    predict_dataset = Dataset.from_pandas(predict_df, preserve_index=False)


    # Preprocessing the dataset
    # Padding strategy
    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["text"],
            padding=False, # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            truncation=True,
        )

    predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on prediction dataset",
        ) 

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )
    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
    predictions = np.argmax(predictions, axis=1)
    
    rel_df["prediction"] = [id2label[label] for label in predictions]
    rel_df

    pred_relations = []
    for i, row in tqdm(rel_df.iterrows()):
        filename = row["filename"]
        concept_text_x = row["concept_text_x"]
        concept_text_y = row["concept_text_y"]
        concept_type_x = row["concept_type_x"]
        concept_type_y = row["concept_type_y"]
        start_word_number_x = row["start_word_number_x"]
        end_word_number_x = row["end_word_number_x"]
        start_word_number_y = row["start_word_number_y"]
        end_word_number_y = row["end_word_number_y"]
        line_number = row["start_line"]
        prediction = row["prediction"]
        if prediction != "Other":
            pred_relations.append({"concept_text_x":concept_text_x, "concept_text_y":concept_text_y, "concept_type_x":concept_type_x, "concept_type_y":concept_type_y, "start_word_number_x":start_word_number_x, "end_word_number_x":end_word_number_x, "start_word_number_y":start_word_number_y, "end_word_number_y":end_word_number_y, "line_number":line_number, "filename":filename, "prediction":prediction})
        


