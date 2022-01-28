from flask import Flask, redirect, url_for, request
from clustering.search import search_query
import json

print("ll")
app = Flask(__name__)

print("ll")

@app.route("/hello")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

print("jj")
@app.route("/indexes/train-index/docs/search", methods=["POST"])
def main():
    request_data = request.get_json()
    print("filters", request_data.get("filters"))
    filters =json.loads(request_data.get("filters"))
    records, count_filtered = search_query(query=request_data["query"], filters=filters, top=request_data["top"])
    assert len(set([record["id"] for record in records])) == len(records), "ids of results are not unique"
    res = {"value": records, "count": count_filtered}
    return json.dumps(res, indent=2)


if __name__ == "__main__":
    print("mmmm")
    app.run(host="localhost", port=5000)
