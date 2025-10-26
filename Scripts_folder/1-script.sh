#!/bin/bash

# Function to check if a command exists
command_exists () {
    type "$1" &> /dev/null ;
}

echo "Starting Nginx installation script..."

# Check for root privileges
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    exit 1
fi

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

# Enable Nginx to start on boot
echo "Enabling Nginx to start on boot..."
if systemctl enable nginx; then
    echo "Nginx enabled to start on boot."
else
    echo "Failed to enable Nginx on boot. Exiting."
    exit 1
fi

# Start Nginx service
echo "Starting Nginx service..."
if systemctl start nginx; then
    echo "Nginx service started successfully."
else
    echo "Failed to start Nginx service. Exiting."
    exit 1
fi

# Verify Nginx status
echo "Verifying Nginx status..."
if systemctl is-active --quiet nginx; then
    echo "Nginx is running and active."
    echo "You can check its status with: sudo systemctl status nginx"
    echo "You can also visit http://localhost in your browser to see the default Nginx page."
else
    echo "Nginx is not running. Please check the logs for errors."
    systemctl status nginx --no-pager
    exit 1
fi

echo "Nginx installation and configuration complete."
