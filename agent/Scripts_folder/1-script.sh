#!/bin/bash

# Check if the script is run as root
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Please use 'sudo bash $0' or 'sudo ./$0'."
  exit 1
fi

echo "Starting Nginx installation..."

# Update package lists
echo "Updating package lists..."
if apt update -y; then
  echo "Package lists updated successfully."
else
  echo "Failed to update package lists. Exiting."
  exit 1
fi

# Install Nginx
echo "Installing Nginx..."
if apt install nginx -y; then
  echo "Nginx installed successfully."
else
  echo "Failed to install Nginx. Exiting."
  exit 1
fi

# Start Nginx service
echo "Starting Nginx service..."
if systemctl start nginx; then
  echo "Nginx service started."
else
  echo "Failed to start Nginx service. Please check logs."
  exit 1
fi

# Enable Nginx to start on boot
echo "Enabling Nginx to start on boot..."
if systemctl enable nginx; then
  echo "Nginx enabled to start on boot."
else
  echo "Failed to enable Nginx to start on boot. Please check logs."
  exit 1
fi

echo "Nginx installation and configuration complete."
echo "You can verify Nginx status with: systemctl status nginx"
echo "You can access Nginx by navigating to your server's IP address in a web browser."
