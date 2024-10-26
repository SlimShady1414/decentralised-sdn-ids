import os
import joblib
from flask import Flask, request, jsonify
import numpy as np
from xgboost import XGBClassifier

app = Flask(__name__)

global_model = {
    'rf': None,  # RandomForest global model
    'xgb': None  # XGBoost global model
}

class FederatedAggregator:
    def __init__(self, model_type, num_classes=10, max_trees=100):
        self.model_type = model_type
        self.local_models = []
        self.max_trees = max_trees  # Maximum number of trees to retain in the global model
        self.num_classes = num_classes  # Total number of expected classes

    def update_local_model(self, model):
        """
        Store the received local model for later aggregation.
        """
        self.local_models.append(model)

    def aggregate_rf(self):
        """
        Incremental learning strategy for RandomForest:
        Add new trees to the global model without losing previous predictive capabilities.
        """
        if len(self.local_models) == 0:
            return None

        if global_model['rf'] is None:
            global_model['rf'] = self.local_models[0]
        else:
            # Combine new decision trees from all local models with the global model
            for model in self.local_models:
                global_model['rf'].estimators_ += model.estimators_

        # Prune the global model's trees if we exceed the maximum allowed number of trees
        if len(global_model['rf'].estimators_) > self.max_trees:
            global_model['rf'].estimators_ = global_model['rf'].estimators_[:self.max_trees]

        return global_model['rf']

    def aggregate_xgb(self):
        """
        Incremental learning strategy for XGBoost:
        Add new boosting trees to the global model without losing previous predictions.
        """
        if len(self.local_models) == 0:
            return None

        if global_model['xgb'] is None:
            # Initialize global XGB with all expected classes, even if some classes aren't in the first local model
            global_model['xgb'] = XGBClassifier(n_estimators=10, use_label_encoder=False, num_class=self.num_classes)
            global_model['xgb'].fit(np.zeros((1, 1)), [0])  # Dummy fit to initialize all classes

        for model in self.local_models:
            # Add new boosting trees from local models to the global model
            global_model['xgb'].get_booster().copy().add(model.get_booster())

        # Limit the number of boosting rounds if it exceeds a threshold
        if len(global_model['xgb'].get_booster().get_dump()) > self.max_trees:
            global_model['xgb'].get_booster().prune(self.max_trees)

        return global_model['xgb']

    def aggregate(self):
        """
        General aggregation method: Determine which model type is being aggregated.
        """
        if self.model_type == 'rf':
            return self.aggregate_rf()
        elif self.model_type == 'xgb':
            return self.aggregate_xgb()
        return None


# Define aggregators for RF and XGB
rf_aggregator = FederatedAggregator('rf')
xgb_aggregator = FederatedAggregator('xgb')


@app.route('/update_model', methods=['POST'])
def update_model():
    """
    Endpoint to receive model updates from the clients.
    """
    try:
        model_type = request.form['model_type']
        model_file = request.files['model']
        model_filename = f"/tmp/{model_type}_model.joblib"

        # Save the received model file
        model_file.save(model_filename)

        # Load the model from the saved file
        local_model = joblib.load(model_filename)

        # Update the appropriate aggregator with the local model
        if model_type == 'rf':
            rf_aggregator.update_local_model(local_model)
        elif model_type == 'xgb':
            xgb_aggregator.update_local_model(local_model)

        return jsonify({'message': f'{model_type} model updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up: remove the temporary model file
        if os.path.exists(model_filename):
            os.remove(model_filename)


@app.route('/aggregate', methods=['GET'])
def aggregate_models():
    """
    Endpoint to trigger model aggregation.
    """
    try:
        # Aggregate the models for RF and XGB
        global_model['rf'] = rf_aggregator.aggregate_rf()
        global_model['xgb'] = xgb_aggregator.aggregate_xgb()

        if global_model['rf'] is not None and global_model['xgb'] is not None:
            return jsonify({'message': 'Models aggregated successfully'}), 200
        else:
            return jsonify({'error': 'Aggregation failed due to missing models'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000)