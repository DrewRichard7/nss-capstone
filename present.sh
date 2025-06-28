#!/bin/bash

# MLB Playoff Prediction - Presentation Launcher
# Makes it easy to run the reveal.js presentation

echo "🏆 MLB Playoff Prediction - Presentation Launcher"
echo "=================================================="

# Check if presentation.html exists
if [ ! -f "presentation.html" ]; then
    echo "❌ Error: presentation.html not found in current directory"
    echo "   Make sure you're running this from the capstone folder"
    exit 1
fi

echo "✅ Found presentation.html"
echo ""

# Function to start local server
start_server() {
    echo "🌐 Starting local web server on port 8080..."
    echo "📊 Your presentation will be available at:"
    echo "   http://localhost:8080/presentation.html"
    echo ""
    echo "🎮 Presentation Controls:"
    echo "   • Arrow keys or Space - Navigate slides"
    echo "   • ESC - Overview mode"
    echo "   • F - Fullscreen"
    echo "   • B - Black screen"
    echo "   • ? - Help menu"
    echo ""
    echo "🛑 Press Ctrl+C to stop the server"
    echo "=================================================="
    echo ""

    # Start Python HTTP server
    python3 -m http.server 8080
}

# Function to open directly in browser
open_direct() {
    echo "🚀 Opening presentation directly in browser..."
    if command -v open >/dev/null 2>&1; then
        # macOS
        open presentation.html
    elif command -v xdg-open >/dev/null 2>&1; then
        # Linux
        xdg-open presentation.html
    elif command -v start >/dev/null 2>&1; then
        # Windows
        start presentation.html
    else
        echo "❌ Could not auto-open browser. Please manually open:"
        echo "   file://$(pwd)/presentation.html"
    fi
}

# Main menu
echo "Choose how to run your presentation:"
echo ""
echo "1) 🌐 Local server (Recommended for presentations)"
echo "2) 🚀 Open directly in browser (Quick preview)"
echo "3) 📋 Show file path only"
echo "4) ❌ Cancel"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        start_server
        ;;
    2)
        open_direct
        ;;
    3)
        echo ""
        echo "📂 Presentation file location:"
        echo "   file://$(pwd)/presentation.html"
        echo ""
        echo "💡 Copy this path to your browser address bar"
        ;;
    4)
        echo "❌ Cancelled"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1-4."
        exit 1
        ;;
esac
