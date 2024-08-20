from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

aggregated_predictions = None

@app.route('/update_predictions', methods=['POST'])
def update_predictions():
    global aggregated_predictions
    received_predictions = np.array(request.json['predictions'])
    if aggregated_predictions is None:
        aggregated_predictions = received_predictions
    else:
        aggregated_predictions = np.mean([aggregated_predictions, received_predictions], axis=0)
    return jsonify({'message': 'Predictions received successfully'}), 200

@app.route('/get_aggregated_predictions', methods=['GET'])
def get_aggregated_predictions():
    global aggregated_predictions
    if aggregated_predictions is None:
        return jsonify({'aggregated_predictions': []}), 200
    return jsonify({'aggregated_predictions': aggregated_predictions.tolist()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
