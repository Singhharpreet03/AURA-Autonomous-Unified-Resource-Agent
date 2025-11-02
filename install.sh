#!/bin/bash


PACKAGE_FILE="./AURA-agent-0.1.0.tar.gz"

# 1. Install Dependencies
sudo apt update
sudo apt install -y python3 python3-pip

# 2. Install the Python Package
echo "Installing Python package..."
sudo python3 -m pip install $PACKAGE_FILE

# 3. Create Secure Directories
echo "Creating secure directories..."
sudo mkdir -p /etc/aura/
sudo mkdir -p /var/agent/scripts/
sudo chown root:root /var/agent/scripts/
sudo chmod 700 /var/agent/scripts/

# # 4. Generate and Store Encryption Key (CRITICAL STEP!) only when running locally without cloud console
# echo "Generating and storing encryption key..."
# sudo python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > /tmp/agent_temp_key
# sudo mv /tmp/agent_temp_key /etc/myagent/secret.key
# sudo chown root:root /etc/myagent/secret.key
# sudo chmod 600 /etc/myagent/secret.key


# 5. Set up systemd Service (Copy your myagent.service file and enable it)
echo "Setting up systemd service (Requires AURA.service file)..."
# Replace with the actual path to your service file
sudo cp AURA.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable AURA.service 
sudo systemctl start AURA.service

echo "Installation complete. AURA Agent service started."
