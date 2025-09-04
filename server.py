#!/usr/bin/env python3
"""
Analyseur de cours intelligent - Version améliorée
- OCR gratuit avec Tesseract
- Hébergement gratuit avec Vercel/Railway
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
            # 1. Conversion en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Réduction du bruit
            denoised = cv2.medianBlur(gray, 5)
            
            # 3. Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 4. Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 5. Morphologie pour nettoyer
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
            except:
                return None

    def extract_text_with_tesseract(self, image_data, language='fra+eng'):
        """Extraction de texte avec Tesseract OCR (gratuit)"""
        try:
            print("🔍 Démarrage OCR Tesseract...")
            
            # Amélioration de l'image
            processed_image = self.enhance_image_for_ocr(image_data)
            
            if processed_image is None:
                # Fallback direct
                image = Image.open(io.BytesIO(image_data))
                processed_image = np.array(image.convert('L'))
            
            # Configuration Tesseract optimisée
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789àáâäèéêëìíîïòóôöùúûüÿçñÀÁÂÄÈÉÊËÌÍÎÏÒÓÔÖÙÚÛÜŸÇÑ.,;:!?()[]{}"\'-+*/%=<>&|@#$€ '
            
            # Extraction du texte
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                lang=language, 
                config=custom_config
            )
            
            # Nettoyage du texte
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
        
        # Corrections communes OCR
        corrections = {
            r'\b[Il1]\b': 'l',
            r'\b0\b': 'o',
            r'\brn\b': 'm',
            r'\bvv\b': 'w',
            r'\|': 'l',
            r'(?<=[a-z])(?=[A-Z])': ' ',  # Ajoute espace entre minuscule et majuscule
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Nettoyage général
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Lignes vides multiples
        
        return text.strip()

    def detect_content_type_auto(self, text):
        """Détection automatique du type de contenu"""
        text_lower = text.lower()
        
        # Mots-clés par domaine
        keywords = {
            'mathematics': ['équation', 'théorème', 'fonction', 'dérivée', 'intégrale', 'matrice', 'vecteur', 'géométrie'],
            'physics': ['force', 'énergie', 'vitesse', 'accélération', 'onde', 'électron', 'atome', 'masse'],
            'chemistry': ['molécule', 'atome', 'réaction', 'acide', 'base', 'élément', 'liaison', 'ph'],
            'biology': ['cellule', 'adn', 'protéine', 'enzyme', 'organisme', 'évolution', 'génétique'],
            'history': ['guerre', 'révolution', 'siècle', 'roi', 'empire', 'bataille', 'traité', 'civilisation'],
            'literature': ['auteur', 'roman', 'poème', 'vers', 'métaphore', 'personnage', 'narration'],
            'philosophy': ['existence', 'conscience', 'morale', 'éthique', 'métaphysique', 'logique'],
            'economics': ['marché', 'prix', 'demande', 'offre', 'inflation', 'monnaie', 'commerce']
        }
        
        scores = {}
        for domain, words in keywords.items():
            score = sum(1 for word in words if word in text_lower)
            scores[domain] = score
        
        if scores:
            detected = max(scores, key=scores.get)
            if scores[detected] > 2:
                return detected
        
        return 'general'

analyzer = CourseAnalyzer()

# APIs IA alternatives gratuites/locales
class AIProvider:
    def __init__(self):
        self.providers = ['mistral', 'huggingface', 'ollama', 'local']

    def query_huggingface(self, text, task_type="summarization"):
        """API Hugging Face gratuite (limitée)"""
        if not HUGGINGFACE_API_KEY:
            return None
            
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        
        # Modèles gratuits selon la tâche
        models = {
            "summarization": "facebook/bart-large-cnn",
            "question-generation": "mrm8488/t5-base-finetuned-question-generation-ap",
            "text-generation": "gpt2"
        }
        
        api_url = f"https://api-inference.huggingface.co/models/{models.get(task_type, 'gpt2')}"
        
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": text[:1000]})
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def query_ollama_local(self, prompt, model="llama3.2:1b"):
        """Ollama local (gratuit, nécessite installation)"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get('response', '')
        except:
            pass
        return None

ai_provider = AIProvider()

# Prompts améliorés avec nouvelles fonctionnalités
ENHANCED_SYSTEM_PROMPT = """
Tu es Lya, une IA pédagogique avancée spécialisée dans l'analyse de cours. 
Tu es capable de créer des ressources d'apprentissage personnalisées et interactives.

Nouvelles capacités:
- Détection automatique du niveau (débutant/intermédiaire/avancé)
- Génération de cartes mentales textuelles
- Création de quiz interactifs avec difficulté progressive
- Suggestions de ressources complémentaires
- Plans de révision personnalisés
- Mnémotechniques créatives
"""

MASTER_ANALYSIS_PROMPT = """
TEXTE À ANALYSER:
---
{text}
---

CONTEXTE DÉTECTÉ: {content_type}
NIVEAU ESTIMÉ: {level}

GÉNÈRE UNE ANALYSE COMPLÈTE avec les sections suivantes (utilise les balises XML exactes):

<section_resume>
## RÉSUMÉ STRUCTURÉ
[Résumé détaillé avec points clés et relations entre concepts]
</section_resume>

<section_niveau>
## ANALYSE DE NIVEAU
[Estime le niveau: débutant/intermédiaire/avancé et explique pourquoi]
</section_niveau>

<section_concepts>
## CONCEPTS CLÉS HIÉRARCHISÉS
[Liste les concepts par ordre d'importance avec définitions]
</section_concepts>

<section_carte_mentale>
## CARTE MENTALE TEXTUELLE
[Structure arborescente des idées principales et leurs connexions]
</section_carte_mentale>

<section_questions_progressives>
## QUIZ PROGRESSIF (3 NIVEAUX)
### Niveau Facile:
[3 questions de base]
### Niveau Moyen:  
[3 questions d'application]
### Niveau Difficile:
[2 questions d'analyse/synthèse]
</section_questions_progressives>

<section_mnémotechniques>
## TECHNIQUES DE MÉMORISATION
[Moyens mnémotechniques, acronymes, associations pour retenir les concepts]
</section_mnémotechniques>

<section_plan_revision>
## PLAN DE RÉVISION PERSONNALISÉ
[Planning sur 7 jours avec objectifs quotidiens]
</section_plan_revision>

<section_ressources>
## RESSOURCES COMPLÉMENTAIRES SUGGÉRÉES
[Types de ressources à chercher pour approfondir]
</section_ressources>

<section_evaluation>
## AUTO-ÉVALUATION
[Grille de compétences à cocher pour mesurer sa compréhension]
</section_evaluation>
"""

def analyze_with_ai(text, content_type="general", detail_level="detailed"):
    """Analyse complète avec IA (plusieurs providers)"""
    
    # Détection automatique du niveau
    level = "intermédiaire"  # Par défaut
    if len(text) < 500:
        level = "débutant"
    elif len(text) > 2000 and any(word in text.lower() for word in ['théorème', 'démonstration', 'corollaire', 'axiome']):
        level = "avancé"
    
    full_prompt = MASTER_ANALYSIS_PROMPT.format(
        text=text, 
        content_type=content_type, 
        level=level
    )
    
    # Essai avec différents providers
    providers = [
        ("Mistral API", lambda: query_mistral_api(ENHANCED_SYSTEM_PROMPT, full_prompt)),
        ("Hugging Face", lambda: ai_provider.query_huggingface(text)),
        ("Ollama Local", lambda: ai_provider.query_ollama_local(full_prompt)),
        ("Analyse locale", lambda: generate_local_analysis(text, content_type, level))
    ]
    
    for provider_name, provider_func in providers:
        try:
            print(f"🤖 Tentative avec {provider_name}...")
            result = provider_func()
            if result and len(result) > 100:
                print(f"✅ Analyse réussie avec {provider_name}")
                return result
        except Exception as e:
            print(f"❌ Échec {provider_name}: {e}")
            continue
    
    # Fallback analysis locale si tout échoue
    return generate_local_analysis(text, content_type, level)

def query_mistral_api(system_prompt, user_prompt):
    """Query Mistral avec gestion d'erreurs améliorée"""
    if not MISTRAL_API_KEY:
        return None
        
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {MISTRAL_API_KEY}'
    }
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        elif response.status_code == 429:
            time.sleep(2)  # Retry après rate limit
            return query_mistral_api(system_prompt, user_prompt)
        else:
            return None
            
    except Exception as e:
        print(f"Erreur Mistral API: {e}")
        return None

def generate_local_analysis(text, content_type, level):
    """Analyse locale de fallback (sans IA externe)"""
    
    # Analyse basique locale
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    # Concepts basés sur la fréquence
    word_freq = {}
    for word in words:
        if len(word) > 4:  # Mots significatifs
            word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
    
    top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return f"""
<section_resume>
## RÉSUMÉ AUTOMATIQUE
Le texte analysé contient {len(words)} mots et {len(sentences)} phrases.
Type de contenu détecté: {content_type}
Niveau estimé: {level}

Contenu principal extrait automatiquement du document fourni.
</section_resume>

<section_concepts>
## CONCEPTS LES PLUS FRÉQUENTS
{chr(10).join([f"**{concept.title()}**: Terme important mentionné {freq} fois" for concept, freq in top_concepts])}
</section_concepts>

<section_questions_progressives>
## QUESTIONS DE RÉVISION AUTO-GÉNÉRÉES
### Niveau Facile:
1. Quels sont les mots-clés principaux de ce cours ?
2. Quel est le thème général abordé ?
3. Combien de concepts principaux sont développés ?

### Niveau Moyen:
1. Comment les différents concepts s'articulent-ils entre eux ?
2. Quelles sont les idées principales à retenir ?
3. Quel est l'objectif pédagogique de ce contenu ?

### Niveau Difficile:
1. Analysez la structure argumentative du document
2. Proposez une synthèse critique du contenu
</section_questions_progressives>

<section_plan_revision>
## PLAN DE RÉVISION SUGGÉRÉ
**Jour 1-2**: Lecture attentive et identification des concepts
**Jour 3-4**: Mémorisation des définitions principales  
**Jour 5-6**: Application et exercices pratiques
**Jour 7**: Révision générale et auto-évaluation
</section_plan_revision>
"""

def parse_ai_response(response):
    """Parse la réponse IA en sections structurées"""
    sections = [
        'resume', 'niveau', 'concepts', 'carte_mentale', 
        'questions_progressives', 'mnémotechniques', 
        'plan_revision', 'ressources', 'evaluation'
    ]
    
    parsed_data = {}
    for section in sections:
        pattern = f'<section_{section}>(.*?)</section_{section}>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            parsed_data[section] = match.group(1).strip()
        else:
            parsed_data[section] = f"Section {section} non générée"
    
    return parsed_data

# Routes API améliorées
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
        if not file.filename:
            return jsonify({'error': 'Nom de fichier vide'}), 400
        
        # Paramètres de la requête
        content_type = request.form.get('content_type', 'general')
        detail_level = request.form.get('detail_level', 'detailed')
        language = request.form.get('language', 'fr')
        
        print(f"🚀 Nouvelle analyse - Type: {content_type}, Langue: {language}")
        
        # Lecture du fichier
        file_data = file.read()
        
        # OCR avec Tesseract
        tesseract_lang = analyzer.languages.get(language, 'fra+eng')
        extracted_text, ocr_error = analyzer.extract_text_with_tesseract(file_data, tesseract_lang)
        
        if ocr_error:
            return jsonify({'error': f"Erreur OCR: {ocr_error}"}), 500
        
        # Auto-détection du type si pas spécifié
        if content_type == 'general':
            content_type = analyzer.detect_content_type_auto(extracted_text)
            print(f"📊 Type auto-détecté: {content_type}")
        
        # Analyse avec IA
        ai_response = analyze_with_ai(extracted_text, content_type, detail_level)
        analysis_data = parse_ai_response(ai_response)
        
        # Stockage en session pour téléchargement
        session_id = f"session_{int(time.time())}"
        session_data[session_id] = {
            'analysis': analysis_data,
            'raw_text': extracted_text,
            'metadata': {
                'content_type': content_type,
                'language': language,
                'processing_time': round(time.time() - start_time, 2),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Génération de flashcards à partir des concepts
        flashcards = generate_flashcards_from_concepts(analysis_data.get('concepts', ''))
        
        print(f"✅ Analyse complète en {round(time.time() - start_time, 2)}s")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'summary': analysis_data.get('resume', ''),
            'level_analysis': analysis_data.get('niveau', ''),
            'concepts': analysis_data.get('concepts', ''),
            'mind_map': analysis_data.get('carte_mentale', ''),
            'progressive_quiz': analysis_data.get('questions_progressives', ''),
            'mnemonics': analysis_data.get('mnémotechniques', ''),
            'study_plan': analysis_data.get('plan_revision', ''),
            'resources': analysis_data.get('ressources', ''),
            'self_evaluation': analysis_data.get('evaluation', ''),
            'flashcards': flashcards,
            'raw_text': extracted_text,
            'metadata': session_data[session_id]['metadata']
        })
        
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Erreur interne du serveur',
            'details': str(e)
        }), 500

def generate_flashcards_from_concepts(concepts_text):
    """Génère des flashcards à partir du texte des concepts"""
    flashcards = []
    
    # Parse les concepts du format "**Concept**: Definition"
    concept_matches = re.findall(r'\*\*([^*]+)\*\*:\s*([^\n]+)', concepts_text)
    
    for concept, definition in concept_matches[:8]:  # Max 8 flashcards
        flashcards.append({
            'question': f"Qu'est-ce que {concept.strip()} ?",
            'answer': definition.strip()
        })
    
    return flashcards

@app.route('/download/<session_id>')
def download_analysis(session_id):
    """Téléchargement des résultats d'analyse"""
    if session_id not in session_data:
        return jsonify({'error': 'Session non trouvée'}), 404
    
    data = session_data[session_id]
    
    # Génération du contenu de téléchargement
    content = generate_download_content(data)
    
    return jsonify({
        'success': True,
        'content': content,
        'filename': f"analyse_cours_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    })

def generate_download_content(session_data):
    """Génère le contenu formaté pour téléchargement"""
    analysis = session_data['analysis']
    metadata = session_data['metadata']
    
    content = f"""
{'='*60}
                    ANALYSE DE COURS INTELLIGENTE
{'='*60}

Générée le: {metadata['timestamp']}
Type de contenu: {metadata['content_type']}
Temps de traitement: {metadata['processing_time']}s

{'='*60}

{analysis.get('resume', '')}

{'='*60}
CONCEPTS CLÉS
{'='*60}
{analysis.get('concepts', '')}

{'='*60}
CARTE MENTALE
{'='*60}
{analysis.get('carte_mentale', '')}

{'='*60}
QUIZ PROGRESSIF
{'='*60}
{analysis.get('questions_progressives', '')}

{'='*60}
PLAN DE RÉVISION
{'='*60}
{analysis.get('plan_revision', '')}

{'='*60}
TEXTE BRUT EXTRAIT (OCR)
{'='*60}
{session_data['raw_text']}
"""
    
    return content

@app.route('/health')
def health_check():
    """Vérification de l'état du service"""
    # Test Tesseract
    tesseract_ok = False
    try:
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except:
        pass
    
    return jsonify({
        'status': 'OK',
        'services': {
            'tesseract_ocr': tesseract_ok,
            'mistral_api': bool(MISTRAL_API_KEY),
            'huggingface_api': bool(HUGGINGFACE_API_KEY),
            'ollama_local': requests.get(f"{OLLAMA_URL}/api/version", timeout=2).status_code == 200 if OLLAMA_URL else False
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
    print("🚀 ANALYSEUR DE COURS INTELLIGENT v2.0")
    print("="*50)
    print("🔍 OCR: Tesseract (gratuit)")
    print("🤖 IA: Multi-provider (Mistral/HuggingFace/Ollama/Local)")
    print("💾 Stockage: En mémoire (session)")
    print("="*50)
    
    # Démarrage selon l'environnement
    if os.getenv('VERCEL'):
        print("☁️ Mode VERCEL")
    elif os.getenv('RAILWAY'):
        print("🚂 Mode RAILWAY") 
    else:
        print("💻 Mode DÉVELOPPEMENT LOCAL")
        print("URL: http://127.0.0.1:5000")
        app.run(host='127.0.0.1', port=5000, debug=True)

