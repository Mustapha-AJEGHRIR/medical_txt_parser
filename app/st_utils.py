import sys
import os

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.parse_data import parse_concept, parse_ast, find_char_indexes
import pandas as pd
from spacy import displacy
from tqdm import tqdm


def visualize_record(record):
    """
    Visualize a single record.
    """
    filename = record["filename"]
    # read file
    with open("data/train/txt" + os.sep + filename + ".txt") as f:
        text = f.read()
    lines = text.split("\n")

    # parse file concepts and assertions
    # assertions_df = pd.DataFrame(parse_ast("data/train/ast/" + filename +".ast"))
    concepts_df = pd.DataFrame(parse_concept("data/train/concept/" + filename + ".con"))
    concepts_df = concepts_df.apply(find_char_indexes, axis=1, args=(text,))
    concepts_df = concepts_df.rename(
        columns={"start_char_index": "start", "end_char_index": "end", "concept_type": "label"}
    )

    ex = [
        {
            "text": line,
            "ents": concepts_df[concepts_df["start_line"] == i + 1][["start", "end", "label"]].to_dict(
                orient="records"
            ),
        }
        for i, line in enumerate(lines)
    ]
    html = displacy.render(
        ex,
        style="ent",
        manual=True,
        jupyter=False,
        options={
            "colors": {
                "problem": "#f08080",
                "treatment": "#9bddff",
                "test": "#ffdab9",
            },
        },
    )

    return html

if __name__ == "__main__":
    visualize_record({"filename": "018636330_DH"})
