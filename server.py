import os
import joblib
from flask import Flask, request, jsonify
import numpy as np
from xgboost import XGBClassifier
from blockchain import Blockchain  # Import Blockchain structure

app = Flask(__name__)

# Define initial model paths
initial_model_paths = {
    'rf': '/mnt/SharedCapstone/rf_model.joblib',
    'xgb': '/mnt/SharedCapstone/xgb_model.joblib',
    'orf': '/mnt/SharedCapstone/orf_model.joblib',
    'ht': '/mnt/SharedCapstone/ht_model.joblib'
}

# Initialize blockchain with the initial pre-trained models as the genesis block
blockchain = Blockchain(initial_model_paths)

# Load initial models into global_model dictionary
global_model = {
    'rf': joblib.load(initial_model_paths['rf']),
    'xgb': joblib.load(initial_model_paths['xgb']),
    'orf': joblib.load(initial_model_paths['orf']),
    'ht': joblib.load(initial_model_paths['ht'])
}

class FederatedAggregator:
    def __init__(self, model_type, num_classes=10, max_trees=100):
        self.model_type = model_type
        self.local_models = []
        self.max_trees = max_trees  # Maximum number of trees to retain in the global model
        self.num_classes = num_classes  # Total number of expected classes

    def update_local_model(self, model):
        """Store the received local model for later aggregation."""
        self.local_models.append(model)

    def aggregate_rf(self):
        """Incremental learning for RandomForest."""
        if len(self.local_models) == 0:
            return None

        if global_model['rf'] is None:
            global_model['rf'] = self.local_models[0]
        else:
            for model in self.local_models:
                global_model['rf'].estimators_ += model.estimators_

        if len(global_model['rf'].estimators_) > self.max_trees:
            global_model['rf'].estimators_ = global_model['rf'].estimators_[:self.max_trees]

        return global_model['rf']

    def aggregate_xgb(self):
        """Incremental learning for XGBoost."""
        if len(self.local_models) == 0:
            return None

        if global_model['xgb'] is None:
            global_model['xgb'] = XGBClassifier(n_estimators=10, use_label_encoder=False, num_class=self.num_classes)
            global_model['xgb'].fit(np.zeros((1, 1)), [0])

        for model in self.local_models:
            global_model['xgb'].get_booster().copy().add(model.get_booster())

        if len(global_model['xgb'].get_booster().get_dump()) > self.max_trees:
            global_model['xgb'].get_booster().prune(self.max_trees)

        return global_model['xgb']

    def aggregate(self):
        if self.model_type == 'rf':
            return self.aggregate_rf()
        elif self.model_type == 'xgb':
            return self.aggregate_xgb()
        return None


# Initialize aggregators for each model type
rf_aggregator = FederatedAggregator('rf')
xgb_aggregator = FederatedAggregator('xgb')
orf_aggregator = FederatedAggregator('orf')
ht_aggregator = FederatedAggregator('ht')

@app.route('/update_model', methods=['POST'])
def update_model():
    """Receive model updates from clients and trigger aggregation."""
    try:
        model_type = request.form['model_type']
        model_file = request.files['model']
        model_filename = f"/mnt/SharedCapstone/{model_type}_model_update.joblib"  # Temporary file for update

        model_file.save(model_filename)
        local_model = joblib.load(model_filename)

        if model_type == 'rf':
            rf_aggregator.update_local_model(local_model)
        elif model_type == 'xgb':
            xgb_aggregator.update_local_model(local_model)
        elif model_type == 'orf':
            orf_aggregator.update_local_model(local_model)
        elif model_type == 'ht':
            ht_aggregator.update_local_model(local_model)

        # Trigger aggregation after model update
        response = aggregate_models()
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(model_filename):
            os.remove(model_filename)  # Clean up

@app.route('/aggregate', methods=['GET'])
def aggregate_models():
    """Trigger model aggregation and add results to the blockchain."""
    try:
        # Paths for new aggregated models
        model_paths = {
            'rf': f"/mnt/SharedCapstone/rf_model_v{len(blockchain.chain)}.joblib",
            'xgb': f"/mnt/SharedCapstone/xgb_model_v{len(blockchain.chain)}.joblib",
            'orf': f"/mnt/SharedCapstone/orf_model_v{len(blockchain.chain)}.joblib",
            'ht': f"/mnt/SharedCapstone/ht_model_v{len(blockchain.chain)}.joblib"
        }

        for model_type, path in model_paths.items():
            aggregator = globals()[f"{model_type}_aggregator"]
            global_model[model_type] = aggregator.aggregate()

            # Save only if aggregation was successful
            if global_model[model_type] is not None:
                joblib.dump(global_model[model_type], path)
            else:
                model_paths[model_type] = initial_model_paths[model_type]

        # Add the block to the blockchain
        new_block = blockchain.add_block(model_paths, f"v{len(blockchain.chain)}")

        # Return block details in the response
        return jsonify({
            'message': 'Models aggregated and stored in blockchain successfully',
            'block_index': new_block.index,
            'block_version': new_block.version,
            'block_hash': new_block.hash
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)
