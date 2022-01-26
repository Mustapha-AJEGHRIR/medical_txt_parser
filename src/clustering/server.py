import sys, os
from flask import Flask, request, redirect, render_template

# import jsonschema
# from jsonschema import validate

# sys.path.append(os.path.dirname(__file__)) #In order to be able to solve relative path problems

from search import search


# valide_json_schema = {
#     "type": "object",
#     "properties": {
#         "ph": {"type": "array"},
#         "Hardness": {"type": "array"},
#         "Solids": {"type": "array"},
#         "Chloramines": {"type": "array"},
#         "Sulfate": {"type": "array"},
#         "Conductivity": {"type": "array"},
#         "Organic_carbon": {"type": "array"},
#         "Trihalomethanes": {"type": "array"},
#         "Turbidity": {"type": "array"},
#     },
#     "additionalProperties": False
# }
# def validateJson(jsonData):
#     try:
#         validate(instance=jsonData, schema=valide_json_schema)
#     except jsonschema.exceptions.ValidationError as err:
#         return False
#     return True


app = Flask(__name__)

@app.route('/search', methods = ["GET"])
def search_query():
    data = dict(request.json)
    # print(data)
    query = data["query"]
    return search(query)


if __name__ == '__main__':
    app.run(debug=True, port=5000)