from flask import Flask, redirect, url_for, request
app = Flask(__name__)

@app.route("/hello")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route("/indexes/train-index/docs/search", methods=['POST'])
def search():
    # data = pd.read_excel('C:/Users/adsieg/Desktop/flask_tuto/wine_filter.xlsx', encoding='utf8')
    # data = data[(data['country']==request.form["Country"]) & (data['province']==request.form["Region"])]
    # data = data.head(50)
    # data = data.to_dict(orient='records')
    # response = json.dumps(data, indent=2)
    request_data = request.get_json()
    print(request_data)
    return {}

if __name__ == "__main__":
    app.run(host='localhost', port=5000)