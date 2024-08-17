import joblib
import numpy as np
import pandas as pd
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp

class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.model = joblib.load('/mnt/SharedCapstone/multi_attack_model.pkl')
        self.scaler = joblib.load('/mnt/SharedCapstone/feature_scaler_all_features.pkl')
        self.features = [
            'Init_Win_bytes_forward', 'Fwd Packet Length Max', 'Fwd Packet Length Mean',
            'Subflow Fwd Bytes', 'Avg Fwd Segment Size', 'Subflow Fwd Packets',
            'Total Length of Fwd Packets', 'Bwd Packet Length Min', 'act_data_pkt_fwd',
            'Fwd IAT Std', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Bwd IAT Mean',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Average Packet Size', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s'
        ]

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

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.logger.info("Switch connected: %s", datapath.id)
        elif ev.state == ofproto_v1_3.OFPPR_DELETE:
            if datapath.id in self.datapaths:
                self.logger.info('Unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)

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

        self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        ipv4_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        # Initialize a list of zeros for all features
        features = [0] * len(self.features)

        if ipv4_pkt:
            features[9] = len(msg.data)  # Flow Duration (example)

            if tcp_pkt:
                features[0] = tcp_pkt.window_size  # Init_Win_bytes_forward
                features[10] = tcp_pkt.seq  # Fwd IAT Std (example)
                # Populate other TCP features similarly
            if udp_pkt:
                features[1] = udp_pkt.total_length  # Fwd Packet Length Max (example)
                # Populate other UDP features similarly

            try:
                # Convert features list to DataFrame with correct feature names
                features_df = pd.DataFrame([features], columns=self.features)
                features_scaled = self.scaler.transform(features_df)
                prediction = self.model.predict(features_scaled)[0]

                attack_labels = {
                    0: 'BENIGN',
                    1: 'DoS Hulk',
                    2: 'PortScan',
                    3: 'DDoS',
                    4: 'DoS GoldenEye',
                    5: 'FTP-Patator',
                    6: 'SSH-Patator',
                    7: 'DoS slowloris',
                    8: 'DoS Slowhttptest',
                    9: 'Bot',
                    10: 'Web Attack – Brute Force',
                    11: 'Web Attack – XSS',
                    12: 'Infiltration',
                    13: 'Web Attack – Sql Injection',
                    14: 'Heartbleed'
                }

                attack_type = attack_labels.get(prediction, 'Unknown')

                if attack_type != 'BENIGN':
                    self.logger.info("Attack detected: %s from %s", attack_type, src)
                else:
                    self.logger.info("Normal traffic detected from %s", src)
            except Exception as e:
                self.logger.error("Error in feature scaling or prediction: %s", e)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
