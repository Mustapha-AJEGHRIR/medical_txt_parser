import sys
import os

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.parse_data import parse_concept, parse_ast, find_char_indexes
import pandas as pd
from tqdm import tqdm
import spacy_streamlit


def visualize_record(record, task="assertion"):
    """
    Visualize a single record.
    task can be "concept" or "assertion"
    """
    assert task in ["concept", "assertion"], "task must be 'concept' or 'assertion'"
    
    filename = record["filename"]
    with open("data/train/txt" + os.sep + filename + ".txt") as f:
        text = f.read()
    lines = text.split("\n")

    # parse file concepts and assertions
    if task == "concept":
        df = pd.DataFrame(parse_concept("data/train/concept/" + filename + ".con"))
        df = df.apply(find_char_indexes, axis=1, args=(text,))
        df = df.rename(columns={"start_char_index": "start", "end_char_index": "end", "concept_type": "label"})
        possible_labels = ["problem", "test", "treatment"]
    elif task == "assertion":
        df = pd.DataFrame(parse_ast("data/train/ast/" + filename + ".ast"))
        df = df.apply(find_char_indexes, axis=1, args=(text,))
        df = df.rename(columns={"start_char_index": "start", "end_char_index": "end", "assertion_type": "label"})
        possible_labels = ["present", "possible", "absent", "conditional", "hypothetical", "associated_with_someone_else"]

    doc = [
        {
            "text": line,
            "ents": df[df["start_line"] == i + 1][["start", "end", "label"]].to_dict(orient="records"),
        }
        for i, line in enumerate(lines)
    ]

    return spacy_streamlit.visualize_ner(
        doc,
        labels=possible_labels,
        show_table=False,
        title="",
        manual=True,
        displacy_options={
            "colors": {
                "problem": "#f08080",
                "treatment": "#9bddff",
                "test": "#ffdab9",
                "present": "#f08080",
                "possible": "#00ffff",
                "absent": "#ff00ff",
                "conditional": "#ffa500",
                "hypothetical": "#ffdab9",
                "associated_with_someone_else": "#00ff7f"
            },
        },
    )


if __name__ == "__main__":
    visualize_record({"filename": "018636330_DH"})
