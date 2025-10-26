#!/bin/bash

# Check if the script is run as root
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Please use sudo."
  exit 1
fi

echo "--- Starting Nginx installation and setup ---"

# Update package lists
echo "1. Updating package lists..."
if apt update -y; then
  echo "Package lists updated successfully."
else
  echo "Error: Failed to update package lists. Exiting."
  exit 1
fi

# Install Nginx
echo "2. Installing Nginx..."
if apt install nginx -y; then
  echo "Nginx installed successfully."
else
  echo "Error: Failed to install Nginx. Exiting."
  exit 1
fi

# Enable Nginx to start on boot
echo "3. Enabling Nginx to start on boot..."
if systemctl enable nginx; then
  echo "Nginx enabled to start on boot."
else
  echo "Warning: Failed to enable Nginx to start on boot. Please check manually."
fi

# Start Nginx service immediately
echo "4. Starting Nginx service..."
if systemctl start nginx; then
  echo "Nginx service started successfully."
else
  echo "Error: Failed to start Nginx service. Exiting."
  exit 1
fi

# Verify Nginx status
echo "5. Verifying Nginx status..."
if systemctl is-active --quiet nginx; then
  echo "Nginx is running and active."
  echo "You can check its full status with: sudo systemctl status nginx"
else
  echo "Error: Nginx is not running. Please check the service status."
  exit 1
fi

echo "--- Nginx installation and setup complete ---"
echo "You can now access your Nginx web server by navigating to your server's IP address or domain name in a web browser."
