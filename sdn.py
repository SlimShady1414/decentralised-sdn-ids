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
        self.model = joblib.load('/mnt/SharedCapstone/ensemble_multi_attack_model.pkl')
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
        self.last_status = None

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

    def extract_features(self, pkt, tcp_pkt, udp_pkt):
        features = [0] * len(self.features)

        if tcp_pkt:
            features[0] = tcp_pkt.window_size
            features[1] = len(pkt)
            features[2] = len(pkt)
            features[3] = len(pkt)
            features[4] = len(pkt)
            features[5] = 1
            features[6] = len(pkt)
            features[7] = min(tcp_pkt.dst_port, tcp_pkt.src_port)
            features[8] = len(pkt)
            features[9] = len(pkt)
            features[10] = len(pkt)
            features[11] = 1
            features[12] = 1
            features[13] = len(pkt)
            features[14] = max(tcp_pkt.dst_port, tcp_pkt.src_port)
            features[15] = min(tcp_pkt.dst_port, tcp_pkt.src_port)
            features[16] = len(pkt)
            features[17] = len(pkt)
            features[18] = len(pkt)
            features[19] = np.std([len(pkt)])
            features[20] = np.var([len(pkt)])
            features[21] = np.mean([len(pkt)])
            features[22] = 1
            features[23] = 1
            features[24] = len(pkt)
            features[25] = len(pkt)
        elif udp_pkt:
            features[0] = udp_pkt.sport
            features[1] = udp_pkt.length
            features[2] = udp_pkt.length
            features[3] = udp_pkt.length
            features[4] = udp_pkt.length
            features[5] = 1
            features[6] = len(pkt)
            features[7] = udp_pkt.sport
            features[8] = 1
            features[9] = len(pkt)
            features[10] = len(pkt)
            features[11] = 1
            features[12] = 1
            features[13] = len(pkt)
            features[14] = udp_pkt.sport
            features[15] = udp_pkt.dport
            features[16] = len(pkt)
            features[17] = len(pkt)
            features[18] = len(pkt)
            features[19] = np.std([len(pkt)])
            features[20] = np.var([len(pkt)])
            features[21] = np.mean([len(pkt)])
            features[22] = 1
            features[23] = 1
            features[24] = len(pkt)
            features[25] = len(pkt)

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

                attack_labels = {
                    0: 'Normal',
                    1: 'Botnet',
                    2: 'Brute Force',
                    3: 'DoS/DDoS',
                    4: 'Infiltration',
                    5: 'Port Scan',
                    6: 'Web Attack'
                }

                attack_type = attack_labels.get(prediction, 'Unknown')

                if attack_type != self.last_status:
                    if attack_type != 'Normal':
                        self.logger.info("Attack detected: %s from %s", attack_type, src)
                    else:
                        self.logger.info("Normal traffic detected from %s", src)
                    self.last_status = attack_type
            except Exception as e:
                self.logger.error("Error in feature scaling or prediction: %s", e)
        else:
            if self.last_status != 'Non-IP or unsupported':
                self.logger.info("Non-IP or unsupported packet type received.")
                self.last_status = 'Non-IP or unsupported'

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
