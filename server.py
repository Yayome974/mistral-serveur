#!/usr/bin/env python3
"""
Analyseur de cours intelligent - Version améliorée
- OCR gratuit avec Tesseract
- Hébergement gratuit avec Render/Vercel/Railway
- Nouvelles fonctionnalités avancées
"""

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
import numpy as np
from dotenv import load_dotenv
import requests
from werkzeug.utils import secure_filename

# Import conditionnel de cv2
try:
    import cv2
    CV2_AVAILABLE = True
    print("✅ OpenCV disponible")
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV non disponible, utilisation de PIL uniquement")

# Configuration pour différents environnements de déploiement
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

# Configuration Tesseract selon l'environnement
if os.getenv('RENDER') or os.getenv('VERCEL') or os.getenv('RAILWAY'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    print("🐧 Environnement Linux détecté")
else:
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        print("🪟 Environnement Windows détecté")
    else:
        print("🐧 Environnement Unix/Linux local")

# APIs
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')

# Stockage en mémoire
session_data = {}

class CourseAnalyzer:
    def __init__(self):
        self.languages = {'fr': 'fra+eng', 'en': 'eng', 'es': 'spa+eng', 'de': 'deu+eng'}

    def enhance_image_for_ocr_opencv(self, image_data):
        """Amélioration d'image avec OpenCV si disponible"""
        if not CV2_AVAILABLE:
            return None
            
        try:
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return None
                
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Débruitage
            denoised = cv2.medianBlur(gray, 3)
            
            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            return enhanced
        except Exception as e:
            print(f"Erreur OpenCV: {e}")
            return None

    def enhance_image_for_ocr_pil(self, image_data):
        """Amélioration d'image avec PIL (fallback)"""
        try:
            # Ouvrir l'image avec PIL
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convertir en niveaux de gris
            gray_image = pil_image.convert('L')
            
            # Améliorer le contraste
            enhanced = ImageEnhance.Contrast(gray_image).enhance(1.5)
            
            # Améliorer la netteté
            enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.2)
            
            return np.array(enhanced)
        except Exception as e:
            print(f"Erreur PIL: {e}")
            return None

    def enhance_image_for_ocr(self, image_data):
        """Amélioration d'image avec fallback automatique"""
        # Essayer OpenCV en premier si disponible
        if CV2_AVAILABLE:
            enhanced = self.enhance_image_for_ocr_opencv(image_data)
            if enhanced is not None:
                print("✅ Image améliorée avec OpenCV")
                return enhanced
        
        # Fallback sur PIL
        enhanced = self.enhance_image_for_ocr_pil(image_data)
        if enhanced is not None:
            print("✅ Image améliorée avec PIL")
            return enhanced
        
        print("❌ Impossible d'améliorer l'image")
        return None

    def extract_text_with_tesseract(self, image_data, language='fra+eng'):
        """Extraction de texte avec Tesseract"""
        try:
            print(f"🔍 Démarrage OCR Tesseract avec langue: {language}")
            
            # Améliorer l'image
            processed_image = self.enhance_image_for_ocr(image_data)
            
            if processed_image is None:
                print("⚠️ Utilisation de l'image brute pour OCR")
                # Fallback : utiliser l'image originale
                pil_image = Image.open(io.BytesIO(image_data))
                processed_image = np.array(pil_image.convert('L'))
            
            # Configuration Tesseract
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ.,?!:;-()[]{}"\' '
            
            # Extraction du texte
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                lang=language, 
                config=custom_config
            )
            
            # Nettoyage du texte
            cleaned_text = self.clean_extracted_text(extracted_text)
            
            if len(cleaned_text.strip()) < 10:
                return None, "Peu de texte détecté. Vérifiez la qualité de l'image."
            
            print(f"✅ OCR réussi: {len(cleaned_text)} caractères extraits")
            return cleaned_text, None
            
        except pytesseract.TesseractNotFoundError:
            error_msg = "Tesseract non trouvé. Vérifiez l'installation."
            print(f"❌ {error_msg}")
            return None, error_msg
        except pytesseract.TesseractError as e:
            error_msg = f"Erreur Tesseract: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
        except Exception as e:
            error_msg = f"Erreur OCR: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg

    def clean_extracted_text(self, text):
        """Nettoyage du texte extrait"""
        if not text:
            return ""
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les lignes trop courtes (probablement du bruit)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 2]
        
        return '\n'.join(cleaned_lines).strip()

    def detect_content_type_auto(self, text):
        """Détection automatique du type de contenu"""
        if not text:
            return 'general'
            
        text_lower = text.lower()
        
        keywords = {
            'mathematics': ['équation', 'théorème', 'fonction', 'dérivée', 'intégrale', 'limite', 'matrice', 'vecteur', 'nombre', 'calcul'],
            'physics': ['force', 'énergie', 'vitesse', 'accélération', 'onde', 'particule', 'champ', 'masse', 'température', 'pression'],
            'chemistry': ['molécule', 'atome', 'réaction', 'acide', 'élément', 'solution', 'ph', 'chimique', 'liaison', 'ion'],
            'biology': ['cellule', 'adn', 'protéine', 'enzyme', 'génétique', 'organisme', 'tissu', 'membrane', 'métabolisme'],
            'history': ['guerre', 'roi', 'siècle', 'bataille', 'empire', 'révolution', 'date', 'période', 'civilisation'],
            'literature': ['poème', 'roman', 'auteur', 'personnage', 'style', 'métaphore', 'récit', 'littéraire'],
            'philosophy': ['concept', 'pensée', 'existence', 'conscience', 'morale', 'éthique', 'être', 'raison'],
        }
        
        scores = {}
        for domain, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            scores[domain] = score
        
        detected = max(scores, key=scores.get) if scores else 'general'
        max_score = scores.get(detected, 0)
        
        # Seuil minimum pour une détection fiable
        if max_score < 2:
            return 'general'
        
        print(f"🎯 Type de contenu détecté: {detected} (score: {max_score})")
        return detected

analyzer = CourseAnalyzer()

# Prompt maître complet
MASTER_ANALYSIS_PROMPT = """
TEXTE À ANALYSER:
---
{text}
---
CONTEXTE: {content_type} | NIVEAU CIBLE: {level}
GÉNÈRE UNE ANALYSE PÉDAGOGIQUE COMPLÈTE. UTILISE LES BALISES XML EXACTES POUR CHAQUE SECTION:

<section_resume>
## RÉSUMÉ STRUCTURÉ
[Résumé clair et concis avec les points clés en gras.]
</section_resume>

<section_niveau>
## ANALYSE DE NIVEAU
[Estime le niveau du contenu (débutant, intermédiaire, avancé) et justifie en 2-3 lignes.]
</section_niveau>

<section_concepts>
## CONCEPTS CLÉS HIÉRARCHISÉS
[Liste les 5 concepts majeurs par ordre d'importance. Pour chaque concept, fournis une définition simple. Format: **Concept**: Définition.]
</section_concepts>

<section_carte_mentale>
## CARTE MENTALE TEXTUELLE
[Crée une carte mentale arborescente simple (utilise des tirets et des indentations) reliant le sujet principal à ses sous-thèmes.]
</section_carte_mentale>

<section_questions_progressives>
## QUIZ PROGRESSIF
### Niveau Facile:
1. [Question de définition]
2. [Question de base]
### Niveau Moyen:
1. [Question d'application]
2. [Question de comparaison]
### Niveau Difficile:
1. [Question d'analyse ou de synthèse]
</section_questions_progressives>

<section_mnemotechniques>
## TECHNIQUES DE MÉMORISATION
[Propose 1 ou 2 moyens mnémotechniques créatifs (acronyme, analogie) pour retenir une information clé.]
</section_mnemotechniques>

<section_plan_revision>
## PLAN DE RÉVISION (3 JOURS)
[Propose un plan simple sur 3 jours. Jour 1: ..., Jour 2: ..., Jour 3: ...]
</section_plan_revision>

<section_ressources>
## RESSOURCES COMPLÉMENTAIRES
[Suggère 2 types de ressources externes (ex: "Vidéos de simulation sur YouTube", "Articles de vulgarisation scientifique").]
</section_ressources>
"""

def query_mistral_api(system_prompt, user_prompt):
    """Requête vers l'API Mistral"""
    if not MISTRAL_API_KEY:
        print("⚠️ Clé API Mistral non configurée")
        return None
        
    try:
        print("🤖 Envoi de la requête à Mistral API...")
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                'Authorization': f'Bearer {MISTRAL_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.5,
                "max_tokens": 2500
            },
            timeout=90
        )
        
        response.raise_for_status()
        result = response.json()['choices'][0]['message']['content'].strip()
        
        print(f"✅ Réponse Mistral reçue: {len(result)} caractères")
        return result
        
    except requests.exceptions.Timeout:
        print("❌ Timeout lors de l'appel à l'API Mistral")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur API Mistral: {e}")
        return None
    except Exception as e:
        print(f"❌ Erreur inattendue Mistral: {e}")
        return None

def create_fallback_analysis(text, content_type):
    """Analyse de fallback si aucune IA n'est disponible"""
    print("🔄 Génération d'analyse de fallback...")
    
    word_count = len(text.split())
    char_count = len(text)
    
    # Estimation du niveau basée sur la longueur
    if char_count < 500:
        level = "débutant"
    elif char_count > 2000:
        level = "avancé"
    else:
        level = "intermédiaire"
    
    # Extraction de mots-clés simples
    words = text.lower().split()
    common_words = set(['le', 'la', 'les', 'de', 'du', 'des', 'et', 'à', 'un', 'une', 'dans', 'pour', 'avec', 'sur'])
    important_words = [word for word in words if len(word) > 4 and word not in common_words]
    key_words = list(set(important_words))[:5]
    
    return f"""
<section_resume>
## RÉSUMÉ STRUCTURÉ
Ce document de {word_count} mots traite du domaine **{content_type}**. Le contenu présente plusieurs concepts et informations structurées de manière {level}.
</section_resume>

<section_niveau>
## ANALYSE DE NIVEAU
Le niveau estimé est **{level}** basé sur la complexité et la longueur du texte ({char_count} caractères).
</section_niveau>

<section_concepts>
## CONCEPTS CLÉS HIÉRARCHISÉS
{chr(10).join([f"**{word.capitalize()}**: Concept important du document" for word in key_words[:5]])}
</section_concepts>

<section_carte_mentale>
## CARTE MENTALE TEXTUELLE
- Sujet principal: {content_type.capitalize()}
  - Concept 1: {key_words[0] if key_words else 'Information'}
  - Concept 2: {key_words[1] if len(key_words) > 1 else 'Détail'}
  - Concept 3: {key_words[2] if len(key_words) > 2 else 'Exemple'}
</section_carte_mentale>

<section_questions_progressives>
## QUIZ PROGRESSIF
### Niveau Facile:
1. Quels sont les éléments principaux de ce document ?
2. À quel domaine appartient ce contenu ?
### Niveau Moyen:
1. Comment les concepts s'articulent-ils entre eux ?
2. Quelles sont les applications pratiques ?
### Niveau Difficile:
1. Analysez les implications de ces concepts dans un contexte plus large.
</section_questions_progressives>

<section_mnemotechniques>
## TECHNIQUES DE MÉMORISATION
Créez un acronyme avec les premières lettres des concepts clés pour faciliter la mémorisation.
</section_mnemotechniques>

<section_plan_revision>
## PLAN DE RÉVISION (3 JOURS)
**Jour 1**: Lecture attentive et identification des concepts clés
**Jour 2**: Création de liens entre les concepts et exercices pratiques
**Jour 3**: Révision générale et auto-évaluation
</section_plan_revision>

<section_ressources>
## RESSOURCES COMPLÉMENTAIRES
- Recherches spécialisées sur le sujet
- Documents académiques complémentaires
</section_ressources>
"""

def analyze_with_ai(text, content_type, detail_level):
    """Analyse avec IA (Mistral en priorité, fallback si nécessaire)"""
    if not text or len(text.strip()) < 10:
        print("❌ Texte trop court pour l'analyse")
        return create_fallback_analysis("Texte insuffisant", content_type)
    
    # Estimation du niveau basée sur la longueur
    char_count = len(text)
    if char_count < 500:
        level = "débutant"
    elif char_count > 2000:
        level = "avancé"
    else:
        level = "intermédiaire"
    
    # Formatage du prompt
    full_prompt = MASTER_ANALYSIS_PROMPT.format(
        text=text[:3000],  # Limiter la taille pour éviter les timeouts
        content_type=content_type,
        level=level
    )
    
    # Tentative avec Mistral API
    result = query_mistral_api(None, full_prompt)
    if result:
        print("✅ Analyse réussie avec Mistral API")
        return result
    
    # Fallback si aucune IA disponible
    print("⚠️ Fallback: génération d'analyse basique")
    return create_fallback_analysis(text, content_type)

def parse_ai_response(response):
    """Parseur pour extraire les sections de la réponse IA"""
    sections = [
        'resume', 'niveau', 'concepts', 'carte_mentale', 
        'questions_progressives', 'mnemotechniques', 
        'plan_revision', 'ressources'
    ]
    
    parsed_data = {}
    
    for section in sections:
        pattern = f'<section_{section}>(.*?)</section_{section}>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            # Nettoyer le contenu
            content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # Supprimer les ## en début de ligne
            parsed_data[section] = content
        else:
            parsed_data[section] = f"Contenu pour '{section}' non généré."
            print(f"⚠️ Section '{section}' non trouvée dans la réponse")
    
    return parsed_data

def generate_flashcards_from_concepts(concepts_text):
    """Génération de flashcards à partir des concepts"""
    if not concepts_text:
        return []
    
    flashcards = []
    
    # Recherche des concepts au format **Concept**: Définition
    matches = re.findall(r'\*\*(.*?)\*\*:\s*(.*?)(?=\*\*|$)', concepts_text, re.DOTALL)
    
    for concept, definition in matches[:8]:  # Limiter à 8 flashcards
        concept = concept.strip()
        definition = definition.strip().replace('\n', ' ')
        
        if concept and definition and len(definition) > 10:
            flashcards.append({
                'question': f"Que signifie '{concept}' ?",
                'answer': definition[:200] + ('...' if len(definition) > 200 else '')
            })
    
    # Si pas assez de flashcards, en créer des génériques
    if len(flashcards) < 3:
        words = concepts_text.split()[:50]  # Prendre les 50 premiers mots
        if words:
            flashcards.append({
                'question': 'Quels sont les concepts principaux de ce cours ?',
                'answer': ' '.join(words[:30])
            })
    
    print(f"📚 {len(flashcards)} flashcards générées")
    return flashcards

# Routes Flask

@app.route('/')
def index():
    """Page d'accueil"""
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint principal d'analyse"""
    try:
        print("🚀 Début de l'analyse")
        
        # Vérification du fichier
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Récupération des paramètres
        content_type = request.form.get('content_type', 'general')
        detail_level = request.form.get('detail_level', 'detailed')
        language = request.form.get('language', 'fr')
        
        print(f"📊 Paramètres: type={content_type}, niveau={detail_level}, langue={language}")
        
        # Lecture du fichier
        file_data = file.read()
        print(f"📁 Fichier lu: {len(file_data)} octets")
        
        # Extraction OCR
        tesseract_language = analyzer.languages.get(language, 'fra+eng')
        extracted_text, ocr_error = analyzer.extract_text_with_tesseract(file_data, tesseract_language)
        
        if ocr_error:
            print(f"❌ Erreur OCR: {ocr_error}")
            return jsonify({'error': f"Erreur OCR: {ocr_error}"}), 500
        
        if not extracted_text or len(extracted_text.strip()) < 5:
            return jsonify({'error': 'Aucun texte exploitable détecté dans l\'image'}), 400
        
        print(f"✅ Texte extrait: {len(extracted_text)} caractères")
        
        # Détection automatique du type de contenu si nécessaire
        if content_type == 'general':
            detected_type = analyzer.detect_content_type_auto(extracted_text)
            content_type = detected_type
        
        # Analyse IA
        ai_response = analyze_with_ai(extracted_text, content_type, detail_level)
        analysis_data = parse_ai_response(ai_response)
        
        # Génération des flashcards
        flashcards = generate_flashcards_from_concepts(analysis_data.get('concepts', ''))
        
        # Stockage en session
        session_id = f"session_{int(time.time())}_{hash(extracted_text) % 10000}"
        session_data[session_id] = {
            'analysis': analysis_data,
            'raw_text': extracted_text,
            'metadata': {
                'content_type': content_type,
                'detail_level': detail_level,
                'language': language,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print(f"✅ Analyse terminée, session: {session_id}")
        
        # Réponse JSON
        return jsonify({
            'success': True,
            'session_id': session_id,
            'summary': analysis_data.get('resume'),
            'level_analysis': analysis_data.get('niveau'),
            'concepts': analysis_data.get('concepts'),
            'mind_map': analysis_data.get('carte_mentale'),
            'progressive_quiz': analysis_data.get('questions_progressives'),
            'mnemonics': analysis_data.get('mnemotechniques'),
            'study_plan': analysis_data.get('plan_revision'),
            'resources': analysis_data.get('ressources'),
            'flashcards': flashcards,
            'raw_text': extracted_text,
            'metadata': {
                'content_type': content_type,
                'text_length': len(extracted_text),
                'processing_time': 'completed'
            }
        })
        
    except Exception as e:
        print(f"❌ Erreur critique dans /analyze: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': 'Erreur interne du serveur',
            'details': str(e) if app.debug else 'Contactez le support'
        }), 500

@app.route('/download/<session_id>')
def download_analysis(session_id):
    """Téléchargement de l'analyse"""
    try:
        if session_id not in session_data:
            return jsonify({'error': 'Session non trouvée'}), 404
        
        data = session_data[session_id]
        analysis = data['analysis']
        raw_text = data['raw_text']
        
        # Formatage du contenu pour téléchargement
        content = f"""
ANALYSE AUTOMATIQUE DE COURS
============================
Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session: {session_id}

TEXTE ORIGINAL EXTRAIT:
{raw_text}

RÉSUMÉ:
{analysis.get('resume', 'Non disponible')}

CONCEPTS CLÉS:
{analysis.get('concepts', 'Non disponible')}

PLAN DE RÉVISION:
{analysis.get('plan_revision', 'Non disponible')}

TECHNIQUES DE MÉMORISATION:
{analysis.get('mnemotechniques', 'Non disponible')}

RESSOURCES COMPLÉMENTAIRES:
{analysis.get('ressources', 'Non disponible')}
"""
        
        filename = f"analyse_cours_{session_id}.txt"
        
        return jsonify({
            'success': True,
            'content': content,
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur de téléchargement: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Vérification de l'état des services"""
    tesseract_ok = False
    mistral_ok = bool(MISTRAL_API_KEY)
    
    # Test Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        tesseract_ok = True
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract non disponible: {e}")
    
    # Test APIs
    if mistral_ok:
        print("✅ Clé API Mistral configurée")
    else:
        print("⚠️ Clé API Mistral non configurée")
    
    return jsonify({
        'status': 'OK',
        'services': {
            'tesseract_ocr': tesseract_ok,
            'mistral_api': mistral_ok,
            'opencv': CV2_AVAILABLE,
        },
        'environment': {
            'python_version': os.sys.version,
            'platform': os.name,
            'render': bool(os.getenv('RENDER')),
        },
        'timestamp': datetime.now().isoformat()
    })

# Gestionnaires d'erreurs

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'Fichier trop volumineux (maximum 5 Mo)'}), 413

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ Erreur 500: {error}")
    return jsonify({'error': 'Erreur interne du serveur'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ressource non trouvée'}), 404

# Point d'entrée

if __name__ == '__main__':
    print("🚀 Démarrage de l'Analyseur de Cours Intelligent")
    print(f"🐍 Python {os.sys.version}")
    print(f"🌍 Environnement: {'Render' if os.getenv('RENDER') else 'Local'}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
