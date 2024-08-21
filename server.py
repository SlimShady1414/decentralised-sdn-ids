from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

global_model_weights = None

@app.route('/update_model', methods=['POST'])
def update_model():
    global global_model_weights
    received_weights = np.array(request.json['weights'])
    if global_model_weights is None:
        global_model_weights = received_weights
    else:
        global_model_weights = np.mean([global_model_weights, received_weights], axis=0)
    return jsonify({'message': 'Model weights updated successfully'}), 200

@app.route('/get_model', methods=['GET'])
def get_model():
    global global_model_weights
    if global_model_weights is None:
        return jsonify({'weights': []}), 200
    return jsonify({'weights': global_model_weights.tolist()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

