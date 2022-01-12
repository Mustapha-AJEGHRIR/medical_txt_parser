import re
import os


def parse_ast(ast_filepath):
    # check if file exists in folder
    if not os.path.isfile(ast_filepath):
        return {}

    result = {}
    with open(ast_filepath, "r") as f:
        lines = f.readlines()
        # remove endlines
        lines = [line.strip() for line in lines]
        # remove empty lines
        lines = [line for line in lines if line.strip()]

        # find regex expression: c="pain" 55:10 55:10\|\|t="problem"\|\|a="hypothetical"
        regex = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"\|\|a="(.*?)"')
        matches = [regex.match(line) for line in lines if regex.match(line)]
        result["concept_text"] = [match.group(1) for match in matches]
        result["start_line"] = [int(match.group(2)) for match in matches]
        result["start_word_number"] = [int(match.group(3)) for match in matches]
        result["end_line"] = [int(match.group(4)) for match in matches]
        result["end_word_number"] = [int(match.group(5)) for match in matches]
        result["concept_type"] = [match.group(6) for match in matches]
        result["assertion_type"] = [match.group(7) for match in matches]

    return result


def parse_concept(concept_filepath):
    # check if file exists in folder
    if not os.path.isfile(concept_filepath):
        return {}

    result = {}
    with open(concept_filepath, "r") as f:
        lines = f.readlines()
        # remove endlines
        lines = [line.strip() for line in lines]
        # remove empty lines
        lines = [line for line in lines if line.strip()]

        # find regex expression: c="a workup" 27:2 27:3\|\|t="test"
        regex = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|t="(.*?)"')
        matches = [regex.match(line) for line in lines if regex.match(line)]
        result["concept_text"] = [match.group(1) for match in matches]
        result["start_line"] = [int(match.group(2)) for match in matches]
        result["start_word_number"] = [int(match.group(3)) for match in matches]
        result["end_line"] = [int(match.group(4)) for match in matches]
        result["end_word_number"] = [int(match.group(5)) for match in matches]
        result["concept_type"] = [match.group(6) for match in matches]
    return result


def parse_relation(relation_filepath):
    # check if file exists in folder
    if not os.path.isfile(relation_filepath):
        return {}

    result = {}
    with open(relation_filepath, "r") as f:
        lines = f.readlines()
        # remove endlines
        lines = [line.strip() for line in lines]
        # remove empty lines
        lines = [line for line in lines if line.strip()]

        # find regex expression: c="percocet" 55:1 55:1\|\|r="TrAP"\|\|c="pain" 55:10 55:10
        regex = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)\|\|r="(.*?)"\|\|c="(.*?)" (\d+):(\d+) (\d+):(\d+)')
        matches = [regex.match(line) for line in lines if regex.match(line)]
        result["concept_text_1"] = [match.group(1) for match in matches]
        result["start_line_1"] = [int(match.group(2)) for match in matches]
        result["start_word_number_1"] = [int(match.group(3)) for match in matches]
        result["end_line_1"] = [int(match.group(4)) for match in matches]
        result["end_word_number_1"] = [int(match.group(5)) for match in matches]
        result["concept_text_2"] = [match.group(7) for match in matches]
        result["start_line_2"] = [int(match.group(8)) for match in matches]
        result["start_word_number_2"] = [int(match.group(9)) for match in matches]
        result["end_line_2"] = [int(match.group(10)) for match in matches]
        result["end_word_number_2"] = [int(match.group(11)) for match in matches]
        result["relation_type"] = [match.group(6) for match in matches]
    return result
