#!/bin/bash

# Start the Flask API in the background
echo "Starting Flask API..."
python src/api.py &

# Give the API a moment to start
sleep 5

# Start Streamlit in the foreground
echo "Starting Streamlit Dashboard..."
streamlit run src/app_streamlit.py --server.port 8501 --server.address 0.0.0.0
