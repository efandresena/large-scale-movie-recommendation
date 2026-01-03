#!/bin/bash

# Quick Start Script for Movie Recommendation System

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        Movie Recommendation System - Quick Start           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python installation
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7 or higher."
    exit 1
fi
echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data models
touch data/.gitkeep models/.gitkeep
echo "✓ Directories created"
echo ""

# Check config
echo "Checking configuration..."
if grep -q "YOUR_GOOGLE_DRIVE_FILE_ID_HERE" config.py; then
    echo "⚠️  WARNING: You need to update the Google Drive file ID in config.py"
    echo "   Please follow the instructions in SETUP_GUIDE.md"
    echo ""
    read -p "Have you updated config.py? (y/n): " updated
    if [ "$updated" != "y" ]; then
        echo "Please update config.py first, then run this script again."
        exit 1
    fi
fi
echo "✓ Configuration OK"
echo ""

# All set!
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                     Setup Complete! ✓                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "To start the application, run:"
echo "  python cli.py"
echo ""
echo "Or with the virtual environment:"
echo "  source venv/bin/activate"
echo "  python cli.py"
echo ""
