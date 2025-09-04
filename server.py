#!/usr/bin/env python3
"""
Analyseur de cours intelligent - Version améliorée
- OCR gratuit avec Tesseract
- Hébergement gratuit avec Vercel/Railway
- Nouvelles fonctionnalités avancées
"""

# NOTE POUR LE DÉVELOPPEUR :
# L'erreur "SyntaxError: default 'except:' must be last" signifie qu'un bloc `try...except`
# dans votre code a un `except:` générique AVANT un `except` spécifique (ex: `except ValueError:`).
# Le `except:` générique doit TOUJOURS être le dernier de la liste.
#
# EXEMPLE INCORRECT :
# try:
#     ...
# except:  # Incorrect, car il est avant un except spécifique
#     ...
# except ValueError:
#     ...
#
# EXEMPLE CORRECT :
# try:
#     ...
# except ValueError: # Le spécifique d'abord
#     ...
# except: # Le générique en dernier
#     ...
#
# Veuillez vérifier tous les blocs `try...except` dans votre code pour trouver cette erreur.
# J'ai amélioré la robustesse de ce fichier pour éviter des erreurs 500,
# notamment dans la fonction `health_check`.

import os
import json
import time
import re
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageEnhance, ImageFilter
import io
import pytesseract
import cv2
import numpy as np
from dotenv import load_dotenv
import requests
from werkzeug.utils import secure_filename

# Configuration pour différents environnements de déploiement
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

# Configuration Tesseract selon l'environnement
if os.getenv('VERCEL') or os.getenv('RAILWAY'):
    # Pour les déploiements cloud
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
else:
    # Pour le développement local (Windows/Mac/Linux)
    if os.name == 'nt':  # Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# APIs gratuites alternatives
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')  # Alternative gratuite
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')  # Local AI

# Stockage en mémoire pour la session (remplace localStorage)
session_data = {}

class CourseAnalyzer:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        self.languages = {
            'fr': 'fra+eng',
            'en': 'eng',
            'es': 'spa+eng',
            'de': 'deu+eng'
        }

    def enhance_image_for_ocr(self, image_data):
        """Améliore la qualité de l'image pour un meilleur OCR"""
        try:
            # Conversion en numpy array pour OpenCV
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                # Fallback avec PIL
                pil_image = Image.open(io.BytesIO(image_data))
                img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Amélioration de l'image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 5)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"Erreur enhancement image: {e}")
            # Fallback simple avec PIL
            try:
                pil_image = Image.open(io.BytesIO(image_data))
                enhanced = ImageEnhance.Contrast(pil_image).enhance(2.0)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.5)
                return np.array(enhanced.convert('L'))
            except Exception:
                return None

    def extract_text_with_tesseract(self, image_data, language='fra+eng'):
        """Extraction de texte avec Tesseract OCR (gratuit)"""
        try:
            print("🔍 Démarrage OCR Tesseract...")
            
            processed_image = self.enhance_image_for_ocr(image_data)
            
            if processed_image is None:
                image = Image.open(io.BytesIO(image_data))
                processed_image = np.array(image.convert('L'))
            
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789àáâäèéêëìíîïòóôöùúûüÿçñÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜŸÇÑ.,;:!?()[]{}"\'-+*/%=<>&|@#$€ '
            
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                lang=language, 
                config=custom_config
            )
            
            cleaned_text = self.clean_extracted_text(extracted_text)
            
            if len(cleaned_text.strip()) < 10:
                return None, "Très peu de texte détecté. Vérifiez la qualité de l'image."
            
            print(f"✅ Texte extrait: {len(cleaned_text)} caractères")
            return cleaned_text, None
            
        except Exception as e:
            return None, f"Erreur OCR Tesseract: {str(e)}"

    def clean_extracted_text(self, text):
        """Nettoie et améliore le texte extrait"""
        if not text:
            return ""
        
        corrections = {
            r'\b[Il1]\b': 'l', r'\b0\b': 'o', r'\brn\b': 'm',
            r'\bvv\b': 'w', r'\|': 'l', r'(?<=[a-z])(?=[A-Z])': ' ',
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()

    def detect_content_type_auto(self, text):
        """Détection automatique du type de contenu"""
        text_lower = text.lower()
        
        keywords = {
            'mathematics': ['équation', 'théorème', 'fonction', 'dérivée', 'intégrale', 'matrice'],
            'physics': ['force', 'énergie', 'vitesse', 'accélération', 'onde', 'électron'],
            'chemistry': ['molécule', 'atome', 'réaction', 'acide', 'base', 'élément'],
            'biology': ['cellule', 'adn', 'protéine', 'enzyme', 'organisme', 'génétique'],
            'history': ['guerre', 'révolution', 'siècle', 'roi', 'empire', 'bataille'],
            'literature': ['auteur', 'roman', 'poème', 'vers', 'métaphore', 'personnage'],
        }
        
        scores = {domain: sum(1 for word in words if word in text_lower) for domain, words in keywords.items()}
        
        if scores:
            detected = max(scores, key=scores.get)
            if scores[detected] > 1:
                return detected
        
        return 'general'

analyzer = CourseAnalyzer()

class AIProvider:
    def query_huggingface(self, text, task_type="summarization"):
        """API Hugging Face gratuite (limitée)"""
        if not HUGGINGFACE_API_KEY:
            return None
            
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        models = {"summarization": "facebook/bart-large-cnn", "text-generation": "gpt2"}
        api_url = f"https://api-inference.huggingface.co/models/{models.get(task_type, 'gpt2')}"
        
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": text[:1000]}, timeout=30)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erreur API HuggingFace: {e}")
        return None

    def query_ollama_local(self, prompt, model="llama3.2:1b"):
        """Ollama local (gratuit, nécessite installation)"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get('response', '')
        except requests.exceptions.RequestException as e:
            print(f"Erreur API Ollama: {e}")
        return None

ai_provider = AIProvider()

ENHANCED_SYSTEM_PROMPT = """
Tu es Lya, une IA pédagogique avancée spécialisée dans l'analyse de cours. 
Tu es capable de créer des ressources d'apprentissage personnalisées et interactives.
"""

MASTER_ANALYSIS_PROMPT = """
TEXTE À ANALYSER:
---
{text}
---
CONTEXTE DÉTECTÉ: {content_type} - NIVEAU ESTIMÉ: {level}
GÉNÈRE UNE ANALYSE COMPLÈTE avec les sections suivantes (utilise les balises XML exactes):
<section_resume>## RÉSUMÉ STRUCTURÉ
[Résumé détaillé avec points clés]</section_resume>
<section_concepts>## CONCEPTS CLÉS
[Liste les concepts par ordre d'importance avec définitions]</section_concepts>
<section_questions_progressives>## QUIZ PROGRESSIF (3 NIVEAUX)
### Niveau Facile:[3 questions de base]
### Niveau Moyen:[3 questions d'application]
### Niveau Difficile:[2 questions d'analyse/synthèse]</section_questions_progressives>
<section_plan_revision>## PLAN DE RÉVISION PERSONNALISÉ
[Planning sur 7 jours avec objectifs quotidiens]</section_plan_revision>
"""

def analyze_with_ai(text, content_type="general", detail_level="detailed"):
    """Analyse complète avec IA (plusieurs providers)"""
    level = "débutant" if len(text) < 500 else "avancé" if len(text) > 2000 else "intermédiaire"
    full_prompt = MASTER_ANALYSIS_PROMPT.format(text=text, content_type=content_type, level=level)
    
    providers = [
        ("Mistral API", lambda: query_mistral_api(ENHANCED_SYSTEM_PROMPT, full_prompt)),
        ("Ollama Local", lambda: ai_provider.query_ollama_local(full_prompt)),
    ]
    
    for provider_name, provider_func in providers:
        try:
            print(f"🤖 Tentative avec {provider_name}...")
            result = provider_func()
            if result and len(result) > 50:
                print(f"✅ Analyse réussie avec {provider_name}")
                return result
        except Exception as e:
            print(f"❌ Échec {provider_name}: {e}")
    
    return generate_local_analysis(text, content_type, level)

def query_mistral_api(system_prompt, user_prompt):
    """Query Mistral avec gestion d'erreurs améliorée"""
    if not MISTRAL_API_KEY:
        return None
        
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {MISTRAL_API_KEY}'}
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0.7, "max_tokens": 4000
    }
    
    try:
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status() # Lève une exception pour les codes 4xx/5xx
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Erreur Mistral API: {e}")
        return None

def generate_local_analysis(text, content_type, level):
    """Analyse locale de fallback (sans IA externe)"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    word_freq = {word.lower(): words.count(word.lower()) for word in words if len(word) > 4}
    top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return f"""
<section_resume>## RÉSUMÉ AUTOMATIQUE
Le texte analysé contient {len(words)} mots. Type de contenu: {content_type}. Niveau estimé: {level}.</section_resume>
<section_concepts>## CONCEPTS FRÉQUENTS
{chr(10).join([f"**{c.title()}**: mentionné {f} fois" for c, f in top_concepts])}</section_concepts>
<section_questions_progressives>## QUESTIONS DE RÉVISION
### Niveau Facile: 1. Quels sont les mots-clés principaux ?</section_questions_progressives>
<section_plan_revision>## PLAN DE RÉVISION
**Jour 1-2**: Lecture et identification des concepts. **Jour 3-4**: Mémorisation. **Jour 5-7**: Application.</section_plan_revision>
"""

def parse_ai_response(response):
    """Parse la réponse IA en sections structurées"""
    sections = ['resume', 'concepts', 'questions_progressives', 'plan_revision']
    parsed_data = {}
    for section in sections:
        pattern = f'<section_{section}>(.*?)</section_{section}>'
        match = re.search(pattern, response, re.DOTALL)
        parsed_data[section] = match.group(1).strip() if match else f"Section '{section}' non générée."
    return parsed_data

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint d'analyse principal amélioré"""
    start_time = time.time()
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        file = request.files['file']
        language = request.form.get('language', 'fr')
        tesseract_lang = analyzer.languages.get(language, 'fra+eng')
        
        extracted_text, ocr_error = analyzer.extract_text_with_tesseract(file.read(), tesseract_lang)
        if ocr_error:
            return jsonify({'error': f"Erreur OCR: {ocr_error}"}), 500
        
        content_type = analyzer.detect_content_type_auto(extracted_text)
        ai_response = analyze_with_ai(extracted_text, content_type)
        analysis_data = parse_ai_response(ai_response)
        
        session_id = f"session_{int(time.time())}"
        session_data[session_id] = {'analysis': analysis_data, 'raw_text': extracted_text}
        
        flashcards = generate_flashcards_from_concepts(analysis_data.get('concepts', ''))
        
        return jsonify({
            'success': True, 'session_id': session_id,
            'summary': analysis_data.get('resume', ''),
            'concepts': analysis_data.get('concepts', ''),
            'progressive_quiz': analysis_data.get('questions_progressives', ''),
            'study_plan': analysis_data.get('plan_revision', ''),
            'flashcards': flashcards, 'raw_text': extracted_text,
        })
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Erreur interne du serveur', 'details': str(e)}), 500

def generate_flashcards_from_concepts(concepts_text):
    """Génère des flashcards à partir du texte des concepts"""
    flashcards = []
    concept_matches = re.findall(r'\*\*([^*]+)\*\*:\s*([^\n]+)', concepts_text)
    for concept, definition in concept_matches[:8]:
        flashcards.append({'question': f"Qu'est-ce que {concept.strip()} ?", 'answer': definition.strip()})
    return flashcards

@app.route('/health')
def health_check():
    """Vérification de l'état du service (plus robuste)"""
    tesseract_ok = False
    try:
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        pass
    
    ollama_ok = False
    if OLLAMA_URL:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/version", timeout=2)
            ollama_ok = response.status_code == 200
        except requests.exceptions.RequestException:
            ollama_ok = False
            
    return jsonify({
        'status': 'OK',
        'services': {
            'tesseract_ocr': tesseract_ok,
            'mistral_api': bool(MISTRAL_API_KEY),
            'huggingface_api': bool(HUGGINGFACE_API_KEY),
            'ollama_local': ollama_ok
        },
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'Fichier trop volumineux (max 5 Mo)'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
