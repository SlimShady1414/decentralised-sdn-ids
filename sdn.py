import joblib
import pandas as pd
import numpy as np
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, tcp, udp, icmp
from collections import defaultdict
import time
from sklearn.linear_model import SGDClassifier
from ryu.lib import hub

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.packet_counts = defaultdict(int)
        self.model = joblib.load('/mnt/shared/ddos_model_all_features.pkl')  # Load the new model trained with all features
        self.feature_scaler = joblib.load('/mnt/shared/feature_scaler_all_features.pkl')  # Load the feature scaler if used
        self.feature_names = [
            'Init_Win_bytes_forward', 'Fwd Packet Length Max', 'Fwd Packet Length Mean',
            'Subflow Fwd Bytes', 'Avg Fwd Segment Size', 'Subflow Fwd Packets',
            'Total Length of Fwd Packets', 'Bwd Packet Length Min', 'act_data_pkt_fwd',
            'Fwd IAT Std', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Bwd IAT Mean',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Average Packet Size', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s'
        ]

        # Load thresholds
        self.thresholds = self.calculate_thresholds('/mnt/shared/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

        self.features_lists = defaultdict(lambda: defaultdict(list))
        self.features = defaultdict(lambda: defaultdict(int))
        self.ddos_packet_counter = 0
        self.probability_threshold = 0.5  # Set initial threshold for probability
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self.monitor)

        # Track the last detected DDoS activity time for each node
        self.last_ddos_activity = defaultdict(lambda: time.time())
        self.ddos_block_duration = 300  # Duration to block a node (in seconds)

    def calculate_thresholds(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['Flow Bytes/s'].fillna(df['Flow Bytes/s'].median(), inplace=True)
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        df.replace([np.inf, -np.inf], np.finfo(np.float32).max, inplace=True)
        max_value = np.finfo(np.float32).max
        df['Flow Bytes/s'] = df['Flow Bytes/s'].clip(lower=-max_value, upper=max_value)
        df['Flow Packets/s'] = df['Flow Packets/s'].clip(lower=-max_value, upper=max_value)

        mean_std_df = df.groupby('Label').agg(['mean', 'std'])
        ddos_stats = mean_std_df.loc[1]

        thresholds = {}
        for feature in self.feature_names:
            thresholds[feature] = ddos_stats[feature]['mean'] + 2 * ddos_stats[feature]['std']

        return thresholds

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Default flow to allow ping and other basic traffic
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
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                    instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes", ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dpid = format(datapath.id, "d").zfill(16)
        self.mac_to_port.setdefault(dpid, {})

        src = eth.src
        dst = eth.dst

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)
        icmp_pkt = pkt.get_protocol(icmp.icmp)

        # Always allow ICMP (ping) traffic
        if icmp_pkt:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)
            data = None
            if msg.buffer_id == ofproto.OFP_NO_BUFFER:
                data = msg.data
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                      in_port=in_port, actions=actions, data=data)
            datapath.send_msg(out)
            return

        current_time = time.time()
        features = self.extract_features(pkt, in_port, current_time)

        features_df = pd.DataFrame([features], columns=self.feature_names)
        scaled_features = self.feature_scaler.transform(features_df)

        decision_value = self.model.decision_function(scaled_features)[0]
        prediction = 1 if decision_value >= self.probability_threshold else 0
        matching_features = self.count_matching_features(features_df)
        self.logger.info(f"Prediction: {prediction}, Matching features: {matching_features}, Decision value: {decision_value}")

        if prediction == 1 and matching_features >= 5:
            self.logger.warning('DDoS attack detected from %s', src)
            self.block_traffic(datapath, src)
            self.last_ddos_activity[src] = current_time
        elif matching_features >= 5:
            self.logger.warning('Possible DDoS detected from %s based on feature matching', src)
            self.block_traffic(datapath, src)
            self.save_to_update_data(features_df, 1)
            self.last_ddos_activity[src] = current_time
        else:
            self.logger.info('No DDoS detected from %s', src)
            if prediction == 1:
                self.save_to_update_data(features_df, 0)
                self.update_model()  # Update model immediately for benign packets predicted as 1

        self.ddos_packet_counter += 1
        if self.ddos_packet_counter >= 25:
            self.update_model()  # Update model for DDoS packets periodically
            self.ddos_packet_counter = 0

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)

            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def update_features(self, in_port, pkt, tcp_pkt, udp_pkt, current_time):
        if in_port not in self.features_lists:
            self.features_lists[in_port] = {
                'fwd_lengths': [],
                'flow_iat': [],
                'last_pkt_time': current_time
            }
            self.features[in_port] = {
                'init_win_bytes_forward': 0,
                'act_data_pkt_fwd': 0
            }
        pkt_length = len(pkt)
        self.features_lists[in_port]['fwd_lengths'].append(pkt_length)
        self.features[in_port]['act_data_pkt_fwd'] += 1
        if tcp_pkt:
            self.features[in_port]['init_win_bytes_forward'] = tcp_pkt.window_size
        iat = current_time - self.features_lists[in_port]['last_pkt_time']
        self.features_lists[in_port]['flow_iat'].append(iat)
        self.features_lists[in_port]['last_pkt_time'] = current_time

    def extract_features(self, pkt, in_port, current_time):
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        self.update_features(in_port, pkt, tcp_pkt, udp_pkt, current_time)

        fwd_lengths = self.features_lists[in_port]['fwd_lengths']
        flow_iat = self.features_lists[in_port]['flow_iat']
        time_diff = current_time - self.features_lists[in_port]['last_pkt_time']
        time_diff = time_diff if time_diff != 0 else 1  # Prevent division by zero

        features_data = [
            self.features[in_port]['init_win_bytes_forward'],
            max(fwd_lengths) if fwd_lengths else 0,
            np.mean(fwd_lengths) if fwd_lengths else 0,
            sum(fwd_lengths),
            np.mean(fwd_lengths) if fwd_lengths else 0,
            len(fwd_lengths),
            sum(fwd_lengths),
            min(fwd_lengths) if fwd_lengths else 0,
            self.features[in_port]['act_data_pkt_fwd'],
            np.std(flow_iat) if flow_iat else 0,
            time_diff,
            len(fwd_lengths),
            len(self.features_lists[in_port]['bwd_lengths']) if 'bwd_lengths' in self.features_lists[in_port] else 0,
            np.mean(flow_iat) if flow_iat else 0,
            max(flow_iat) if flow_iat else 0,
            min(flow_iat) if flow_iat else 0,
            np.mean(flow_iat) if flow_iat else 0,
            np.mean(self.features_lists[in_port]['bwd_lengths']) if 'bwd_lengths' in self.features_lists[in_port] else 0,
            np.mean(fwd_lengths) if fwd_lengths else 0,
            np.std(fwd_lengths) if fwd_lengths else 0,
            np.var(fwd_lengths) if fwd_lengths else 0,
            np.mean(fwd_lengths) if fwd_lengths else 0,
            self.features[in_port]['act_data_pkt_fwd'],
            self.features[in_port]['act_data_pkt_fwd'],
            len(fwd_lengths) / time_diff if fwd_lengths else 0,
            len(self.features_lists[in_port]['bwd_lengths']) / time_diff if 'bwd_lengths' in self.features_lists[in_port] else 0
        ]

        features_data = [f if not isinstance(f, list) else 0 for f in features_data]
        features_data = [f if not isinstance(f, np.float64) else f.item() for f in features_data]

        print(f"Extracted features: {features_data}")
        return features_data

    def count_matching_features(self, features_df):
        matching_count = 0
        for feature, threshold in self.thresholds.items():
            if features_df.iloc[0][feature] >= threshold:
                matching_count += 1

        return matching_count

    def block_traffic(self, datapath, mac):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_src=mac)
        actions = []
        self.add_flow(datapath, 1, match, actions)
        self.logger.info('Blocking traffic from %s', mac)

    def save_to_update_data(self, features_df, label):
        features_df['Label'] = label
        with open('/mnt/shared/update_data.csv', 'a') as f:
            features_df.to_csv(f, header=f.tell()==0, index=False)
        self.logger.info(f'Saved new data for model update: {features_df}')

    def update_model(self):
        # Load the new data
        new_data = pd.read_csv('/mnt/shared/update_data.csv')
        X_new = new_data[self.feature_names]
        y_new = new_data['Label']

        # Scale the new data
        X_new_scaled = self.feature_scaler.transform(X_new)

        # Update the model
        self.model.partial_fit(X_new_scaled, y_new, classes=[0, 1])

        # Save the updated model
        joblib.dump(self.model, '/mnt/shared/ddos_model_all_features.pkl')
        self.logger.info('Model updated with new data.')

    def monitor(self):
        while True:
            self.logger.info("Monitoring network status...")
            for dp in self.datapaths.values():
                self.send_port_stats_request(dp)
            self.check_and_reinstate_nodes()
            hub.sleep(10)

    def send_port_stats_request(self, datapath):
        self.logger.debug('send_port_stats_request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def check_and_reinstate_nodes(self):
        current_time = time.time()
        for src, last_activity in list(self.last_ddos_activity.items()):
            if current_time - last_activity > self.ddos_block_duration:
                self.logger.info(f'Reinstating node {src} due to inactivity of DDoS activity for {self.ddos_block_duration} seconds.')
                self.reinstate_traffic(src)
                del self.last_ddos_activity[src]

    def reinstate_traffic(self, mac):
        for datapath in self.datapaths.values():
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(eth_src=mac)
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
            self.add_flow(datapath, 1, match, actions)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == ofproto_v1_3.OFPPR_DELETE:
            if datapath.id in self.datapaths:
                self.logger.info('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        self.logger.info('datapath         port     '
                         'rx-pkts  rx-bytes rx-error '
                         'tx-pkts  tx-bytes tx-error')
        self.logger.info('---------------- -------- '
                         '-------- -------- -------- '
                         '-------- -------- --------')
        for stat in sorted(body, key=lambda x: x.port_no):
            self.logger.info('%016x %8x %8d %8d %8d %8d %8d %8d',
                             ev.msg.datapath.id, stat.port_no,
                             stat.rx_packets, stat.rx_bytes, stat.rx_errors,
                             stat.tx_packets, stat.tx_bytes, stat.tx_errors)