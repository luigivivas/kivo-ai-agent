#!/bin/sh
set -e

python ingest.py

exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
