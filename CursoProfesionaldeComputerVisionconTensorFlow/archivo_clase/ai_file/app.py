
from flask import Flask, request, jsonify
from prediction import smartcities

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def parse_request():
    request_data = request.get_json()
    videoBase64 = request_data['video']
    sc = smartcities()
    response_64 = sc.predict(videoBase64)
    return jsonify(output=response_64)

if __name__ == "__main__":
    ##app.run(debug=False, host='0.0.0.0', port= int(os.environ.get("PORT", 8000)))
    app.run(debug=True, host='0.0.0.0', port= 8000)
