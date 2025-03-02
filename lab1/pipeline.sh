#!/bin/bash

sudo apt-get update

sudo apt-get install -y python3 python3-pip

pip3 install numpy pandas scikit-learn

mkdir -p train test

if [ ! -f train/train_data_1.csv ]; then
  echo "Creating data..."
  python3 data_creation.py
else
  echo "Data already exists. Skipping creation."
fi

echo "Preprocessing data..."
python3 data_preprocessing.py

echo "Training model..."
python3 model_preparation.py

echo "Testing model..."
python3 model_testing.py