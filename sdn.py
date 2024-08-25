import sys
import joblib
import numpy as np
import pandas as pd
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
import requests
import time

# Increase the recursion limit to avoid RecursionError
sys.setrecursionlimit(10000)

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.model = joblib.load('/mnt/SharedCapstone/global_model.pkl')
        self.scaler = joblib.load('/mnt/SharedCapstone/feature_scaler_all_features.pkl')
        self.features = [
            'Fwd Header Length', 'Fwd Packet Length Std', 'Bwd Packets/s', 'Fwd Packet Length Mean',
            'Bwd Header Length', 'Fwd IAT Mean', 'Packet Length Std', 'Flow IAT Std',
            'Min Packet Length', 'Fwd Packet Length Min', 'Avg Fwd Segment Size', 'Max Packet Length',
            'ACK Flag Count', 'Packet Length Variance', 'Packet Length Mean', 'Bwd Packet Length Max',
            'Bwd IAT Std', 'Flow IAT Mean', 'Fwd Packet Length Std', 'Bwd IAT Mean',
            'Average Packet Size', 'Bwd IAT Total'
        ]
        self.local_training_data = []
        self.training_interval = 300  # Train every 5 minutes
        self.last_trained = time.time()

    def update_model(self):
        try:
            response = requests.get('http://10.0.2.15:5000/get_model')
            if response.status_code == 200:
                global_model_weights = np.array(response.json()['weights'])
                for i, estimator in enumerate(self.model.estimators_):
                    if hasattr(estimator, 'coef_'):
                        estimator.coef_ = global_model_weights[i]
                    elif hasattr(estimator, 'feature_importances_'):
                        estimator.feature_importances_ = global_model_weights[i]
                self.logger.info("Local model updated with global weights.")
            else:
                self.logger.error("Failed to fetch the global model")
        except Exception as e:
            self.logger.error(f"Failed to update the local model: {e}")

    def train_local_model(self):
        try:
            if len(self.local_training_data) > 0:
                # Create a DataFrame from local training data
                local_df = pd.DataFrame(self.local_training_data, columns=self.features + ['Label'])

                X_local = local_df[self.features]
                y_local = local_df['Label']

                # Scaling features
                X_local_scaled = self.scaler.transform(X_local)

                # Update the model locally
                self.model.fit(X_local_scaled, y_local)

                # Send updated weights to the global server
                weights = [estimator.coef_ for estimator in self.model.estimators_ if hasattr(estimator, 'coef_')]
                requests.post('http://10.0.2.15:5000/update_model', json={'weights': weights})
                self.logger.info("Local model trained and updated on the global server.")

                # Clear local training data after training
                self.local_training_data = []
            else:
                self.logger.info("No new data to train the local model.")
        except Exception as e:
            self.logger.error(f"Error during local model training: {e}")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id, priority=priority,
                                    match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    def extract_features(self, pkt, tcp_pkt, udp_pkt):
        features = [0] * len(self.features)  # Ensure the list matches the feature length

        if tcp_pkt:
            features[0] = tcp_pkt.window_size  # Fwd Header Length
            features[1] = np.std([len(pkt)])   # Fwd Packet Length Std
            features[2] = 1  # Bwd Packets/s (placeholder)
            features[3] = len(pkt)  # Fwd Packet Length Mean
            features[4] = 1  # Bwd Header Length (placeholder)
            features[5] = np.mean([len(pkt)])  # Fwd IAT Mean (placeholder)
            features[6] = np.std([len(pkt)])   # Packet Length Std
            features[7] = np.std([len(pkt)])   # Flow IAT Std (placeholder)
            features[8] = min(len(pkt), tcp_pkt.dst_port)  # Min Packet Length
            features[9] = len(pkt)  # Fwd Packet Length Min
            features[10] = len(pkt)  # Avg Fwd Segment Size (placeholder)
            features[11] = max(len(pkt), tcp_pkt.dst_port)  # Max Packet Length
            features[12] = 1  # ACK Flag Count (placeholder)
            features[13] = np.var([len(pkt)])  # Packet Length Variance
            features[14] = np.mean([len(pkt)])  # Packet Length Mean
            features[15] = max(len(pkt), tcp_pkt.dst_port)  # Bwd Packet Length Max (placeholder)
            features[16] = np.std([len(pkt)])   # Bwd IAT Std (placeholder)
            features[17] = np.mean([len(pkt)])  # Flow IAT Mean (placeholder)
            features[18] = np.std([len(pkt)])   # Fwd Packet Length Std (duplicate, could change)
            features[19] = np.mean([len(pkt)])  # Bwd IAT Mean (placeholder)
            features[20] = np.mean([len(pkt)])  # Average Packet Size (placeholder)
            features[21] = len(pkt)  # Bwd IAT Total (placeholder)

        elif udp_pkt:
            features[0] = udp_pkt.sport  # Fwd Header Length
            features[1] = np.std([len(pkt)])  # Fwd Packet Length Std
            features[2] = 1  # Bwd Packets/s (placeholder)
            features[3] = udp_pkt.length  # Fwd Packet Length Mean
            features[4] = 1  # Bwd Header Length (placeholder)
            features[5] = np.mean([len(pkt)])  # Fwd IAT Mean (placeholder)
            features[6] = np.std([len(pkt)])  # Packet Length Std
            features[7] = np.std([len(pkt)])  # Flow IAT Std (placeholder)
            features[8] = min(len(pkt), udp_pkt.dport)  # Min Packet Length
            features[9] = udp_pkt.length  # Fwd Packet Length Min
            features[10] = udp_pkt.length  # Avg Fwd Segment Size (placeholder)
            features[11] = max(len(pkt), udp_pkt.dport)  # Max Packet Length
            features[12] = 1  # ACK Flag Count (placeholder)
            features[13] = np.var([len(pkt)])  # Packet Length Variance
            features[14] = np.mean([len(pkt)])  # Packet Length Mean
            features[15] = max(len(pkt), udp_pkt.dport)  # Bwd Packet Length Max (placeholder)
            features[16] = np.std([len(pkt)])  # Bwd IAT Std (placeholder)
            features[17] = np.mean([len(pkt)])  # Flow IAT Mean (placeholder)
            features[18] = np.std([len(pkt)])  # Fwd Packet Length Std (duplicate, could change)
            features[19] = np.mean([len(pkt)])  # Bwd IAT Mean (placeholder)
            features[20] = np.mean([len(pkt)])  # Average Packet Size (placeholder)
            features[21] = udp_pkt.length  # Bwd IAT Total (placeholder)

        return features

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        if tcp_pkt or udp_pkt:
            features = self.extract_features(pkt, tcp_pkt, udp_pkt)

            try:
                features_df = pd.DataFrame([features], columns=self.features)
                features_scaled = self.scaler.transform(features_df)
                prediction = self.model.predict(features_scaled)[0]

                if prediction != 0:  # Non-Normal Traffic
                    self.logger.info("Attack detected from %s", src)
                else:
                    self.logger.info("Normal traffic detected from %s", src)

                # Store data for local training
                self.local_training_data.append(features + ['Attack' if prediction != 0 else 'Normal'])

                # Train the local model periodically
                if time.time() - self.last_trained > self.training_interval:
                    self.train_local_model()
                    self.last_trained = time.time()

            except Exception as e:
                self.logger.error("Error in feature scaling or prediction: %s", e)
        else:
            self.logger.info("Normal traffic detected from %s", src)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
