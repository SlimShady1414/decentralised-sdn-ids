import hashlib
import time
import json
import joblib
import os

class Block:
    def __init__(self, index, model_paths, previous_hash, version):
        self.index = index
        self.timestamp = time.time()
        self.model_paths = model_paths
        self.previous_hash = previous_hash
        self.version = version
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        """Calculate the hash of the block."""
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.model_paths, sort_keys=True)}{self.previous_hash}{self.version}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __repr__(self):
        return (f"Block(index={self.index}, version={self.version}, hash={self.hash}, "
                f"previous_hash={self.previous_hash}, timestamp={self.timestamp})")

class Blockchain:
    def __init__(self, initial_models):
        self.chain = []
        self.create_genesis_block(initial_models)

    def create_genesis_block(self, model_paths):
        """Create the genesis block using initial models."""
        genesis_block = Block(0, model_paths, "0", "v0")
        self.chain.append(genesis_block)
        print(f"Genesis block created with hash: {genesis_block.hash}")

    def add_block(self, model_paths, version):
        """Add a new block to the blockchain with updated models."""
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), model_paths, previous_block.hash, version)
        self.chain.append(new_block)
        print(f"Block {new_block.index} added to the blockchain with version {version} and hash {new_block.hash}")
        return new_block

    def get_chain(self):
        """Retrieve the full blockchain as a list of dictionaries for easy JSON serialization."""
        return [{
            "index": block.index,
            "timestamp": block.timestamp,
            "model_paths": block.model_paths,
            "previous_hash": block.previous_hash,
            "version": block.version,
            "hash": block.hash
        } for block in self.chain]

    def display_chain(self):
        """Display all blocks in the blockchain."""
        for block in self.chain:
            print(block)

