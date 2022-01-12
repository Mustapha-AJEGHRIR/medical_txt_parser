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

        # find regex expression c="CONCEPT" num1:num2 num3:num4
        regex = re.compile(r'c="(.*?)" (\d+):(\d+) (\d+):(\d+)')
        matches = [regex.match(line) for line in lines if regex.match(line)]
        result["concept_text"] = [match.group(1) for match in matches]
        result["start_line"] = [int(match.group(2)) for match in matches]
        result["start_word_number"] = [int(match.group(3)) for match in matches]
        result["end_line"] = [int(match.group(4)) for match in matches]
        result["end_word_number"] = [int(match.group(5)) for match in matches]

        # find regex expression t="PROBLEM"
        regex = re.compile(r't="(.*?)"')
        matches = [regex.findall(line) for line in lines]
        result["concept_type"] = [match[0] for match in matches]

        # find regex expression a="ASSERTION"
        regex = re.compile(r'a="(.*?)"')
        matches = [regex.findall(line) for line in lines]
        result["assertion_type"] = [match[0] for match in matches]

    return result
