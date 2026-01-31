import requests
import tensorflow as tf
import numpy as np
import os
import pickle
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
import time

# Configuration
SERVER_URL = "http://localhost:8000"
CLIENT_ID = "client_" + str(int(time.time()))
LOCAL_DATA_DIR = "data"

def download_global_model():
    print(f"[{CLIENT_ID}] Downloading global model...")
    response = requests.get(f"{SERVER_URL}/model")
    if response.status_code == 200:
        with open("client_model.h5", "wb") as f:
            f.write(response.content)
        print(f"[{CLIENT_ID}] Model downloaded.")
        return True
    else:
        print(f"[{CLIENT_ID}] Failed to download model: {response.status_code}")
        return False

def download_public_key():
    print(f"[{CLIENT_ID}] Downloading public key...")
    response = requests.get(f"{SERVER_URL}/public_key")
    if response.status_code == 200:
        with open("server_public_key.pem", "wb") as f:
            f.write(response.content)
        print(f"[{CLIENT_ID}] Public key downloaded.")
        return True
    else:
        print(f"[{CLIENT_ID}] Failed to download public key: {response.status_code}")
        return False

def train_local_model():
    print(f"[{CLIENT_ID}] Training local model...")
    model = tf.keras.models.load_model("client_model.h5")
    
    # Simulate local data (random for demo, ensuring privacy)
    # In a real app, this would load images from LOCAL_DATA_DIR
    x_train = np.random.random((32, 128, 128, 3))
    y_train = np.random.randint(0, 4, 32)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, verbose=1)
    
    return model.get_weights()

def encrypt_update(weights):
    print(f"[{CLIENT_ID}] Encrypting update...")
    
    # We need the encrypt_payload logic here. 
    # To keep client standalone, we'll inline a simplified version or import if available.
    # ideally we would import from a shared lib, but for this demo I will reimplement briefly
    # to avoid path issues if client assumes it's on a different machine.
    
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as sym_padding
    
    with open("server_public_key.pem", "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read())
        
    aes_key = os.urandom(32)
    iv = os.urandom(16)
    
    data = {'client_id': CLIENT_ID, 'weights': weights}
    pickled_data = pickle.dumps(data)
    
    padder = sym_padding.PKCS7(128).padder()
    padded_data = padder.update(pickled_data) + padder.finalize()
    
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    
    package = {
        'key': encrypted_key,
        'iv': iv,
        'data': encrypted_data
    }
    
    return pickle.dumps(package)

def upload_update(encrypted_payload):
    print(f"[{CLIENT_ID}] Uploading encrypted update...")
    files = {'file': ('update.bin', encrypted_payload)}
    response = requests.post(f"{SERVER_URL}/update", files=files)
    print(f"[{CLIENT_ID}] Server Response: {response.json()}")

def main():
    if download_global_model() and download_public_key():
        weights = train_local_model()
        encrypted_payload = encrypt_update(weights)
        upload_update(encrypted_payload)

if __name__ == "__main__":
    main()
