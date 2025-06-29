#!/bin/bash

# SecPrompt Installation Script

echo "🛡️  Installing SecPrompt - Prompt Injection Security Framework"
echo "================================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version is installed, but Python $required_version or higher is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Install core dependencies
echo "Installing core dependencies..."
pip3 install numpy scikit-learn pandas joblib

# Install optional dependencies
echo "Installing optional dependencies..."
pip3 install streamlit plotly

# Install development dependencies (optional)
read -p "Do you want to install development dependencies? (y/n): " install_dev
if [[ $install_dev =~ ^[Yy]$ ]]; then
    echo "Installing development dependencies..."
    pip3 install pytest pytest-cov black flake8 mypy jupyter
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data models prompts notebooks tests

# Test installation
echo "Testing installation..."
python3 test_basic.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "Quick start:"
    echo "  python3 main.py --help                    # Show available commands"
    echo "  python3 main.py generate --size 10        # Generate test payloads"
    echo "  python3 main.py detect --text 'test'      # Test detection"
    echo "  streamlit run dashboard/app.py            # Launch dashboard"
    echo ""
    echo "For more information, see README.md"
else
    echo "❌ Installation test failed. Please check the error messages above."
    exit 1
fi 