# Securing SDNs with Decentralized Intrusion Detection using Federated Learning

This project involves implementing a novel intrusion detection system using advanced technologies such as machine learning, online learning and blockchain into IDS frameworks. Using these technologies, the project aims to detect threats and mitigate false positives while tackling against zero-day attacks.

## Requirements
- Linux Operating System
- Python 3.9
- Mininet
- Ryu
- IPFS

The dataset used in the project is the [CIC-IDS 2018](https://www.unb.ca/cic/datasets/ids-2018.html) dataset.
## Setup Instructions
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.
### 1. Install Ryu and Mininet
Install Ryu:

```bash
sudo apt-get install ryu
pip install ryu
```
Install Mininet:
```
sudo apt-get install mininet
```
Clone the Repository
```
git clone git@github.com:SlimShady1414/decentralised-sdn-ids.git
cd decentralised-sdn-ids
```
### Install Required Python Packages
Create a virtual environment using venv
```
python3.9 -m venv venv
```
Activate the virtual environment
```
source venv/bin/activate
```


-> requirements.txt


```bash
pandas
scikit-learn
xgboost
imbalanced-learn
joblib
river
flask
requests
numpy
ryu
scipy
jsonschema
```
Install the dependencies using the following commands
```bash
pip install -r requirements.txt
```
## Usage
Open a terminal and run the "training.py" code and save the initial models.
```bash
python training.py
```
Open another terminal and start the server using the command:
```bash
python server.py
```
Open another terminal and start the SDN controller using the command:
```bash
ryu-manager sdn.py
```
Open a new terminal and setup the mininet topology using the command:
```bash
sudo mn --topo=single,"n" --mac --switch=ovsk --controller=remote

replace "n" with the number of nodes you want in the network.
```
Once the network is set-up, run any of the attack scripts using the following commands:
```bash
mininet > h1 python attack_simulation.py

mininet > h1 python zeroday_attack.py
```

You can view the logs on the SDN terminal classifying if the network traffic is benign or malicious and model updates on the server terminal along with the blockchain.
