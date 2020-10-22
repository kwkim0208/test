from generate_molecules import prediction
from flask import Flask
from flask import request, jsonify
import time

app = Flask(__name__)

# GET 요청으로 사용(기존 POST)
@app.route("/predict", methods=["POST","GET"])
def predict():
        start = time.time()
        text = {"content" : request.args.get('content')}
        try:
            json_results = prediction(request.args.get('content'))
        except:
            end = time.time()
            return False
        end = time.time()
        return json_results
        
if __name__ == '__main__':
    PORT = 50051	
    #     app.run(host="0.0.0.0", debug=True, port=PORT)
    app.run(host="0.0.0.0", debug=True, port=PORT)
