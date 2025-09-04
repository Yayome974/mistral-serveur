#!/bin/bash
# Script de construction pour Render

echo "🔧 Installation des dépendances Python..."
pip install -r requirements.txt

echo "📦 Mise à jour des paquets système..."
apt-get update

echo "🔤 Installation de Tesseract OCR..."
apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng tesseract-ocr-spa tesseract-ocr-deu

echo "🧪 Test de Tesseract..."
tesseract --version

echo "✅ Construction terminée avec succès!"
