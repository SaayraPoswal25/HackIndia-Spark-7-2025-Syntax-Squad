import numpy as np
import pickle
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from typing import List, Any
import os

def aggregate_weights(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Aggregates weights from multiple clients using Federated Averaging (FedAvg).
    
    Args:
        weights_list: A list of weights, where each item is a list of numpy arrays (layers).
        
    Returns:
        A list of numpy arrays representing the averaged weights.
    """
    if not weights_list:
        return []
    
    # Initialize with the first client's weights
    new_weights = [np.zeros_like(w) for w in weights_list[0]]
    
    # Sum up all weights
    for client_weights in weights_list:
        for i, layer_weights in enumerate(client_weights):
            new_weights[i] += layer_weights
            
    # Divide by number of clients to get average
    num_clients = len(weights_list)
    avg_weights = [w / num_clients for w in new_weights]
    
    return avg_weights

def load_private_key(path: str = "keys/private_key.pem"):
    """Loads the server's private key for decryption."""
    with open(path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None
        )
    return private_key

def decrypt_payload(encrypted_payload: bytes, private_key) -> Any:
    """
    Decrypts the payload received from the client.
    The payload is expected to be a pickled object encrypted with the public key.
    
    NOTE: For large payloads (like model weights), hybrid encryption (AES + RSA) is usually used.
    For this demo, we will assume the client sends an AES key encrypted with RSA, 
    and the data encrypted with AES.
    
    However, to keep it simple but functional for the demo:
    We will use a hybrid approach in this function if the data is large.
    
    Let's assume the payload is a dictionary:
    {
        "key": <AES key encrypted with RSA>,
        "iv": <AES IV>,
        "data": <Data encrypted with AES>
    }
    """
    # For now, let's implement the standard hybrid decryption
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    
    # Deserialize the JSON-like structure (or pickle)
    package = pickle.loads(encrypted_payload)
    
    encrypted_key = package['key']
    iv = package['iv']
    encrypted_data = package['data']
    
    # 1. Decrypt the AES key using RSA
    aes_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # 2. Decrypt the data using AES
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted_data_padded = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # 3. Unpad the data (PKCS7)
    from cryptography.hazmat.primitives import padding as sym_padding
    unpadder = sym_padding.PKCS7(128).unpadder()
    try:
        decrypted_data = unpadder.update(decrypted_data_padded) + unpadder.finalize()
    except ValueError:
        # Fallback for unpadded data
        decrypted_data = decrypted_data_padded

    return pickle.loads(decrypted_data)

def generate_keys(path: str = "keys"):
    """Generates RSA key pair for the server."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # Save private key
    with open(f"{path}/private_key.pem", "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
        
    # Save public key
    public_key = private_key.public_key()
    with open(f"{path}/public_key.pem", "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

def encrypt_payload(data: Any, public_key_path: str = "keys/public_key.pem") -> bytes:
    """
    Encrypts data to be sent to the server using Hybrid Encryption (RSA + AES).
    """
    # Load public key
    with open(public_key_path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())

    # Generate random AES key and IV
    aes_key = os.urandom(32)
    iv = os.urandom(16)
    
    # Encrypt the data with AES
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as sym_padding
    
    # Serialize data
    pickled_data = pickle.dumps(data)
    
    # Pad data
    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(pickled_data) + padder.finalize()
    
    # Encrypt
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    # Encrypt the AES key with RSA
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    # Package everything
    package = {
        'key': encrypted_key,
        'iv': iv,
        'data': encrypted_data
    }
    
    return pickle.dumps(package)

