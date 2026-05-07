# ============================================
# SQL INJECTION DETECTION MIDDLEWARE API
# Complete Flask Application with Interceptor
# Filename: sql_injection_middleware.py
# ============================================

import os
import re
import json
import pickle
from datetime import datetime
from functools import wraps
from urllib.parse import unquote

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, g, make_response
from flask_cors import CORS
from werkzeug.datastructures import MultiDict

# Import the trained model components (from your main script)
# In production, you'd load the saved model
import joblib

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Configuration settings for the SQL Injection Detection Middleware"""

    # API Settings
    API_TITLE = "SQL Injection Detection Middleware"
    API_VERSION = "1.0.0"
    DEBUG_MODE = True

    # Security Settings
    BLOCK_MALICIOUS_REQUESTS = True  # Default to blocking so the middleware actively protects routes
    LOG_SUSPICIOUS_QUERIES = True
    MAX_QUERY_LENGTH = 5000

    # Detection Threshold
    CONFIDENCE_THRESHOLD = 0.6  # Above this is considered malicious

    # Routes to monitor (empty list means monitor all routes)
    MONITORED_ROUTES = []  # e.g., ['/api/users', '/api/products']
    EXCLUDED_ROUTES = ['/health', '/metrics', '/detect_single']  # Don't monitor these

    # Database-like storage for logs (in production, use real database)
    LOG_FILE = 'sql_injection_logs.json'
    MODEL_BUNDLE_PATH = 'best_sql_injection_model.pkl'


app = Flask(__name__)
app.config.from_object(Config)
CORS(app)  # Enable CORS for testing from different origins

# ============================================
# MODEL LOADING AND PREPROCESSING
# ============================================

class SQLInjectionPredictor:
    """
    Wrapper for the trained SQL injection detection model.
    This loads your best trained model and provides prediction functionality.
    """

    def __init__(self, model_path=None):
        """
        Initialize the predictor with the trained model.

        Args:
            model_path: Path to saved model file (if None, uses in-memory model)
        """
        self.model = None
        self.model_type = None
        self.tfidf_vectorizer = None
        self.seq_tokenizer = None
        self.num_scaler = None
        self.feature_columns = None
        self.max_sequence_length = 300
        self.threshold = Config.CONFIDENCE_THRESHOLD
        self.keras_model_path = None
        self.model_loaded = False

        # Try to load saved model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠ No saved model found. Please train the model first.")
            print("  The API will run in simulation mode for testing.")

    def load_model(self, model_path):
        """
        Load the trained model bundle and preprocessing components.

        The training script stores classical ML models directly inside the
        bundle. For Keras-based models the bundle stores the path to the
        separately saved `.keras` model file.
        """
        try:
            # Load the complete pipeline
            pipeline = joblib.load(model_path)
            self.model = pipeline['model']
            self.model_type = pipeline['model_type']
            self.model_name = pipeline.get('model_name', self.model_type)
            self.threshold = pipeline.get('threshold', Config.CONFIDENCE_THRESHOLD)
            self.tfidf_vectorizer = pipeline.get('tfidf_vectorizer')
            self.seq_tokenizer = pipeline.get('seq_tokenizer')
            self.num_scaler = pipeline.get('num_scaler')
            self.feature_columns = pipeline.get('feature_columns', [])
            self.max_sequence_length = pipeline.get('max_sequence_length', 300)
            self.keras_model_path = pipeline.get('keras_model_path')

            if self.model is None and self.keras_model_path:
                self.model = tf.keras.models.load_model(self.keras_model_path)

            self.model_loaded = True
            print(f"✓ Model loaded successfully: {self.model_name} ({self.model_type})")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model_loaded = False

    def _extract_single_query_features(self, sql_query):
        """
        Recreate the same engineered numeric features used in task1.py.

        The middleware must mirror the training-time feature extraction exactly.
        Otherwise the saved scaler and the trained model would receive a feature
        vector with the wrong shape or wrong ordering.
        """
        raw_query = "" if sql_query is None else str(sql_query)
        clean_query = self._clean_sql_query(raw_query)

        sql_keywords = ['union', 'select', 'insert', 'update', 'delete', 'drop', 'create',
                        'alter', 'exec', 'execute', 'sleep', 'benchmark', 'pg_sleep',
                        'waitfor', 'delay', 'having', 'where', 'order by', 'group by']
        special_chars = ["'", '"', ';', '--', '#', '/*', '*/', '||', '&&', '=', '>', '<']

        feature_dict = {
            'query_length': len(clean_query),
            'word_count': len(clean_query.split()),
        }

        for keyword in sql_keywords:
            feature_dict[f'has_{keyword.replace(" ", "_")}'] = 1 if keyword in clean_query else 0

        for char in special_chars:
            feature_name = f'count_{char.replace("*", "star").replace("/", "slash")}'
            feature_dict[feature_name] = raw_query.count(char)

        feature_dict['has_comment'] = 1 if any(c in raw_query for c in ['--', '#', '/*']) else 0
        feature_dict['has_multiple_queries'] = 1 if raw_query.count(';') > 1 else 0
        feature_dict['has_union'] = 1 if 'union' in clean_query else 0
        feature_dict['has_sleep'] = 1 if any(s in clean_query for s in ['sleep', 'benchmark', 'waitfor']) else 0
        feature_dict['or_count'] = clean_query.count(' or ')
        feature_dict['and_count'] = clean_query.count(' and ')
        feature_dict['not_count'] = clean_query.count(' not ')
        feature_dict['single_quote_count'] = raw_query.count("'")
        feature_dict['double_quote_count'] = raw_query.count('"')
        feature_dict['unbalanced_quotes'] = int(
            (feature_dict['single_quote_count'] % 2 == 1) or
            (feature_dict['double_quote_count'] % 2 == 1)
        )

        return clean_query, feature_dict

    def preprocess_query(self, sql_query):
        """
        Preprocess a SQL query for prediction.
        This mirrors the preprocessing done during training.
        """
        if not self.model_loaded:
            # Return placeholder features for simulation mode
            return np.zeros((1, 100))

        # Clean the query (same as training)
        cleaned_query, numeric_feature_dict = self._extract_single_query_features(sql_query)

        # Extract features based on model type
        if self.model_type in ['lstm', 'gru'] and self.seq_tokenizer:
            # Sequence-based features
            sequence = self.seq_tokenizer.texts_to_sequences([cleaned_query])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            features = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post', truncating='post')
            return features
        else:
            # Classical ML and DNN use TF-IDF plus engineered numeric features.
            if self.tfidf_vectorizer:
                tfidf_features = self.tfidf_vectorizer.transform([cleaned_query]).toarray()

                if self.num_scaler is not None and self.feature_columns:
                    numeric_values = [[numeric_feature_dict.get(col, 0) for col in self.feature_columns]]
                    numeric_features = self.num_scaler.transform(numeric_values)
                    return np.hstack([tfidf_features, numeric_features])

                return tfidf_features

            return np.zeros((1, 100))

    def _clean_sql_query(self, sql_query):
        """Clean and normalize SQL query"""
        if not sql_query:
            return ""

        query = str(sql_query).lower().strip()
        query = ' '.join(query.split())

        # Normalize numbers and strings
        import re
        query = re.sub(r'\b\d+\b', 'N', query)
        query = re.sub(r"'[^']*'", "'S'", query)
        query = re.sub(r'"[^"]*"', '"S"', query)

        return query

    def _to_native_prediction(self, is_malicious, confidence, prediction_score, attack_type, explanation):
        """
        Convert NumPy and TensorFlow scalar outputs into plain Python values so
        Flask can serialize the response safely.
        """
        return {
            'is_malicious': bool(is_malicious),
            'confidence': float(round(float(confidence), 4)),
            'prediction_score': float(round(float(prediction_score), 4)),
            'attack_type': attack_type,
            'explanation': explanation
        }

    def predict(self, sql_query):
        """
        Predict if a SQL query contains SQL injection.

        Returns:
            dict: {
                'is_malicious': bool,
                'confidence': float,
                'attack_type': str (if available),
                'explanation': str
            }
        """
        if not sql_query or len(sql_query) > Config.MAX_QUERY_LENGTH:
            return self._to_native_prediction(False, 0.0, 0.0, None, 'Empty or oversized query')

        # If model not loaded, use rule-based detection for testing
        if not self.model_loaded:
            return self._rule_based_detection(sql_query)

        try:
            # Preprocess and predict
            features = self.preprocess_query(sql_query)

            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features)[0]
                is_malicious = prediction_proba[1] > self.threshold
                confidence = prediction_proba[1] if is_malicious else prediction_proba[0]
                prediction_score = prediction_proba[1]
            else:
                if self.model_type in ['dnn', 'lstm', 'gru']:
                    prediction = float(np.asarray(self.model.predict(features, verbose=0)).ravel()[0])
                else:
                    prediction = float(np.asarray(self.model.predict(features)).ravel()[0])
                is_malicious = prediction > self.threshold
                confidence = prediction if is_malicious else 1 - prediction
                prediction_score = prediction

            # Determine attack type based on patterns
            attack_type = self._detect_attack_type(sql_query, prediction_score)

            # Generate explanation
            explanation = self._generate_explanation(sql_query, is_malicious, confidence, attack_type)

            return self._to_native_prediction(
                is_malicious=is_malicious,
                confidence=confidence,
                prediction_score=prediction_score,
                attack_type=attack_type,
                explanation=explanation
            )

        except Exception as e:
            print(f"Prediction error: {e}")
            return self._to_native_prediction(False, 0.0, 0.0, None, f'Prediction error: {str(e)}')

    def _rule_based_detection(self, sql_query):
        """
        Fallback rule-based detection when ML model is not available.
        This is for testing the API functionality.
        """
        query_lower = sql_query.lower()

        # SQL injection patterns
        injection_patterns = {
            'Union-based': r'(\s+union\s+select\s+)',
            'Error-based': r'(convert\(|cast\(|@@version)',
            'Boolean-based': r'(\' or \'1\'=\'1|\" or \"1\"=\"1| or 1=1| and 1=1)',
            'Time-based': r'(sleep\(|benchmark\(|pg_sleep\(|waitfor\s+delay)',
            'Stack-queries': r'(;\s*(drop|delete|insert|update|create|alter))',
            'Meta-based': r'(information_schema|sys\.|master\.)'
        }

        detected_types = []
        max_confidence = 0.0

        for attack_type, pattern in injection_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                detected_types.append(attack_type)
                max_confidence = max(max_confidence, 0.85)

        # Check for quotes and comments
        if "'" in sql_query or '"' in sql_query:
            quote_count = sql_query.count("'") + sql_query.count('"')
            if quote_count % 2 == 1:  # Unbalanced quotes
                max_confidence = max(max_confidence, 0.65)
                detected_types.append('Suspicious quotes')

        if '--' in sql_query or '#' in sql_query or '/*' in sql_query:
            max_confidence = max(max_confidence, 0.75)
            detected_types.append('SQL comment injection')

        is_malicious = len(detected_types) > 0 or max_confidence > 0.6

        attack_type = detected_types[0] if detected_types else None

        score = round(max_confidence, 4) if max_confidence > 0 else 0.5
        return self._to_native_prediction(
            is_malicious=is_malicious,
            confidence=score,
            prediction_score=score,
            attack_type=attack_type,
            explanation=f"Rule-based detection: {', '.join(detected_types) if detected_types else 'No injection patterns found'}"
        )

    def _detect_attack_type(self, sql_query, prediction_score):
        """Detect the specific type of SQL injection attack"""
        query_lower = sql_query.lower()

        attack_indicators = {
            'Union-based': ['union select', 'union all select'],
            'Error-based': ['convert(', 'cast(', '@@version', 'floor(', 'extractvalue', 'updatexml', 'xpath'],
            'Boolean-based': [' or 1=1', " or '1'='1", ' and 1=1', " and '1'='1", "' or '", '" or "'],
            'Time-based': ['sleep(', 'benchmark(', 'pg_sleep(', 'waitfor delay'],
            'Stack-queries-based': ['; drop', '; delete', '; insert', '; update', '; alter', '; create', '; truncate'],
            'Meta-based': ['information_schema', 'sys.tables', 'master..', 'sys.', 'master.']
        }

        for attack_type, indicators in attack_indicators.items():
            for indicator in indicators:
                if indicator in query_lower:
                    return attack_type

        if '--' in query_lower or '#' in query_lower or '/*' in query_lower:
            return 'Comment-based'

        if ';' in query_lower:
            return 'Stack-queries-based'

        if (' or ' in query_lower or ' and ' in query_lower) and ('=' in query_lower or "'" in query_lower or '"' in query_lower):
            return 'Boolean-based'

        return 'Suspicious SQL Pattern' if prediction_score > 0.5 else None

    def _generate_explanation(self, sql_query, is_malicious, confidence, attack_type):
        """Generate human-readable explanation of the detection"""
        if not is_malicious:
            return "No SQL injection patterns detected in this query."

        explanations = []

        if attack_type:
            explanations.append(f"Detected {attack_type} SQL injection pattern")

        # Add specific pattern matches
        query_lower = sql_query.lower()

        if 'union select' in query_lower:
            explanations.append("UNION SELECT pattern detected - common for data extraction")
        if 'or 1=1' in query_lower or "or '1'='1" in query_lower:
            explanations.append("Boolean tautology pattern detected - attempts to bypass authentication")
        if 'sleep(' in query_lower or 'benchmark(' in query_lower:
            explanations.append("Time-based delay function detected - used for blind injection")
        if ';' in sql_query and any(x in query_lower for x in ['drop', 'delete', 'insert']):
            explanations.append("Stacked query detected - multiple SQL statements")
        if '--' in sql_query or '#' in sql_query:
            explanations.append("SQL comment detected - may be attempting to truncate query")

        if confidence > 0.9:
            explanations.append(f"High confidence detection ({confidence*100:.0f}%)")
        elif confidence > 0.7:
            explanations.append(f"Medium confidence detection ({confidence*100:.0f}%)")

        return " | ".join(explanations) if explanations else "Suspicious SQL pattern detected"


# Initialize the predictor
# By default the middleware loads the bundle produced by task1.py.
# If the bundle is missing the API still starts, but falls back to the
# rule-based detector so the routes can still be demonstrated.
predictor = SQLInjectionPredictor(model_path=Config.MODEL_BUNDLE_PATH)

# ============================================
# REQUEST LOGGING AND MIDDLEWARE
# ============================================

class SQLInjectionLogger:
    """Logs and tracks SQL injection attempts"""

    def __init__(self, log_file='sql_injection_logs.json'):
        self.log_file = log_file
        self.attempts = []
        self.load_logs()

    def load_logs(self):
        """Load existing logs from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.attempts = json.load(f)
            except:
                self.attempts = []

    def save_logs(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.attempts, f, indent=2, default=str)

    def log_attempt(self, request_data, prediction, route, method, ip_address):
        """Log a request attempt"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'route': route,
            'method': method,
            'ip_address': ip_address,
            'request_data': request_data,
            'prediction': prediction,
            'is_malicious': prediction.get('is_malicious', False),
            'confidence': prediction.get('confidence', 0),
            'attack_type': prediction.get('attack_type'),
            'blocked': Config.BLOCK_MALICIOUS_REQUESTS and prediction.get('is_malicious', False)
        }

        self.attempts.append(log_entry)

        # Keep only last 10000 logs
        if len(self.attempts) > 10000:
            self.attempts = self.attempts[-10000:]

        if Config.LOG_SUSPICIOUS_QUERIES and prediction.get('is_malicious', False):
            print(f"⚠ SQL INJECTION DETECTED - Route: {route}, Type: {prediction.get('attack_type')}")

        self.save_logs()

    def get_stats(self):
        """Get statistics about logged attempts"""
        if not self.attempts:
            return {'total_attempts': 0, 'malicious_count': 0}

        total = len(self.attempts)
        malicious = sum(1 for a in self.attempts if a.get('is_malicious', False))

        # Count by attack type
        attack_counts = {}
        for attempt in self.attempts:
            attack_type = attempt.get('attack_type')
            if attack_type and attack_type != 'Unknown':
                attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1

        return {
            'total_attempts': total,
            'malicious_count': malicious,
            'benign_count': total - malicious,
            'blocked_count': sum(1 for a in self.attempts if a.get('blocked', False)),
            'attack_type_distribution': attack_counts
        }


# Initialize logger
logger = SQLInjectionLogger()

# ============================================
# MIDDLEWARE - SQL INJECTION INTERCEPTOR
# ============================================

def extract_sql_from_request():
    """
    Extract all SQL-like content from the request.
    Scans query parameters, path parameters, form data, and JSON body.

    Headers are intentionally excluded from the default scan because values
    such as browser user agents are long, noisy strings that were not part of
    the training data and can create false positives for benign requests.
    """
    sql_candidates = []

    # Check query parameters
    for key, value in request.args.items():
        if value and isinstance(value, str):
            sql_candidates.append({'source': f'query_param_{key}', 'value': value})

    # Check path parameters so route segments like /api/products/1' OR '1'='1
    # are inspected before the route logic uses them.
    if request.view_args:
        for key, value in request.view_args.items():
            if value is not None:
                sql_candidates.append({'source': f'path_param_{key}', 'value': str(value)})

    # Check JSON body
    if request.is_json:
        data = request.get_json(silent=True)
        if data:
            def extract_from_dict(d, prefix=''):
                if isinstance(d, dict):
                    for k, v in d.items():
                        extract_from_dict(v, f"{prefix}.{k}" if prefix else k)
                elif isinstance(d, list):
                    for i, item in enumerate(d):
                        extract_from_dict(item, f"{prefix}[{i}]")
                elif isinstance(d, str) and len(d) > 5:
                    sql_candidates.append({'source': f'json_body_{prefix}', 'value': d})
            extract_from_dict(data)

    # Check form data
    for key, value in request.form.items():
        if value and isinstance(value, str) and len(value) > 5:
            sql_candidates.append({'source': f'form_{key}', 'value': value})

    return sql_candidates


def is_sql_query(text):
    """
    Heuristic to determine if a string looks like an SQL query.
    """
    if not text or len(text) < 3:
        return False

    text_lower = text.lower()
    sql_keywords = ['select', 'insert', 'update', 'delete', 'drop', 'create',
                    'alter', 'union', 'where', 'from', 'join', 'table', 'database']

    keyword_count = sum(1 for keyword in sql_keywords if keyword in text_lower)

    # Also check for SQL syntax patterns
    has_sql_patterns = any([
        '=' in text and ("'" in text or '"' in text),
        '--' in text or '#' in text,
        ';' in text,
        keyword_count >= 2
    ])

    obvious_injection_patterns = any([
        ' or ' in text_lower and '=' in text_lower,
        ' and ' in text_lower and '=' in text_lower,
        '--' in text_lower or '#' in text_lower or '/*' in text_lower,
        ';' in text_lower,
        'union select' in text_lower,
        'sleep(' in text_lower or 'benchmark(' in text_lower or 'pg_sleep(' in text_lower,
        '@@version' in text_lower or 'information_schema' in text_lower
    ])

    return (keyword_count >= 1 and has_sql_patterns) or obvious_injection_patterns


def sql_injection_middleware(func):
    """
    Decorator that intercepts requests and checks for SQL injection.
    Use on any route that should be protected.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        g.sql_injection_detected = False
        g.sql_injection_confidence = 0.0
        g.sql_injection_details = []

        # Skip monitoring for excluded routes
        if request.path in Config.EXCLUDED_ROUTES:
            return func(*args, **kwargs)

        # Check if route should be monitored
        if Config.MONITORED_ROUTES and request.path not in Config.MONITORED_ROUTES:
            return func(*args, **kwargs)

        # Extract all potential SQL content from request
        sql_candidates = extract_sql_from_request()

        # Analyze each candidate
        detection_results = []
        is_malicious = False
        highest_confidence = 0.0
        worst_attack = None

        for candidate in sql_candidates:
            candidate_text = candidate['value']

            # Only evaluate values that actually resemble SQL or a common SQLi
            # payload. This avoids sending benign strings such as usernames or
            # browser-like text into the model unnecessarily.
            if not is_sql_query(candidate_text):
                continue

            # Predict using the model
            prediction = predictor.predict(candidate_text)
            candidate['prediction'] = prediction

            detection_results.append(candidate)

            if prediction['is_malicious']:
                is_malicious = True
                if prediction['confidence'] > highest_confidence:
                    highest_confidence = prediction['confidence']
                    worst_attack = prediction.get('attack_type')

        log_data = {
            'candidate_count': len(sql_candidates),
            'sql_like_candidate_count': len(detection_results),
            'candidates': [
                {'source': r['source'], 'value_preview': r['value'][:100]}
                for r in detection_results
            ]
        }

        prediction_summary = {
            'is_malicious': bool(is_malicious),
            'confidence': float(highest_confidence),
            'attack_type': worst_attack,
            'details': [r.get('prediction', {}) for r in detection_results if r.get('prediction')]
        }

        logger.log_attempt(
            request_data=log_data,
            prediction=prediction_summary,
            route=request.path,
            method=request.method,
            ip_address=request.remote_addr
        )

        # Store detection results in Flask's g object for access in route
        g.sql_injection_detected = is_malicious
        g.sql_injection_confidence = highest_confidence
        g.sql_injection_details = detection_results

        # Block if configured and malicious
        if Config.BLOCK_MALICIOUS_REQUESTS and is_malicious:
            return jsonify({
                'error': 'SQL Injection Detected',
                'message': 'Your request has been blocked due to potential SQL injection',
                'confidence': highest_confidence,
                'attack_type': worst_attack,
                'timestamp': datetime.now().isoformat()
            }), 403

        return func(*args, **kwargs)

    return wrapper


# ============================================
# TEST ROUTES FOR POSTMAN TESTING
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    stats = logger.get_stats()
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'model_type': predictor.model_type,
        'blocking_enabled': Config.BLOCK_MALICIOUS_REQUESTS,
        'stats': stats,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/users', methods=['GET', 'POST', 'PUT', 'DELETE'])
@sql_injection_middleware  # Protected by middleware
def users_endpoint():
    """
    Test endpoint for user operations.
    Try different SQL injection payloads in query parameters or body.
    """
    detection_info = {
        'sql_injection_detected': getattr(g, 'sql_injection_detected', False),
        'confidence': getattr(g, 'sql_injection_confidence', 0),
        'inspected': getattr(g, 'sql_injection_details', [])
    }

    response = {
        'endpoint': '/api/users',
        'method': request.method,
        'message': 'User operation completed',
        'sql_injection_check': detection_info
    }

    # Add request data to response for testing visibility
    if request.args:
        response['query_params'] = dict(request.args)
    if request.is_json:
        response['json_body'] = request.get_json()
    if request.form:
        response['form_data'] = dict(request.form)

    if detection_info['sql_injection_detected']:
        response['warning'] = f"SQL injection pattern detected with {detection_info['confidence']*100:.1f}% confidence"

    return jsonify(response)


@app.route('/api/products/<product_id>', methods=['GET', 'PUT', 'DELETE'])
@sql_injection_middleware
def products_endpoint(product_id):
    """
    Test endpoint with path parameter.
    Try SQL injection in product_id (e.g., /api/products/1' OR '1'='1)
    """
    detection_info = {
        'sql_injection_detected': getattr(g, 'sql_injection_detected', False),
        'confidence': getattr(g, 'sql_injection_confidence', 0)
    }

    response = {
        'endpoint': '/api/products',
        'product_id': product_id,
        'method': request.method,
        'sql_injection_check': detection_info
    }

    if request.args:
        response['query_params'] = dict(request.args)

    if detection_info['sql_injection_detected']:
        response['warning'] = f"SQL injection detected in path parameter!"

    return jsonify(response)


@app.route('/api/search', methods=['GET'])
@sql_injection_middleware
def search_endpoint():
    """
    Search endpoint - common target for SQL injection.
    Try: ?q=' OR '1'='1&category=1 UNION SELECT * FROM users
    """
    detection_info = {
        'sql_injection_detected': getattr(g, 'sql_injection_detected', False),
        'confidence': getattr(g, 'sql_injection_confidence', 0),
        'details': getattr(g, 'sql_injection_details', [])
    }

    response = {
        'endpoint': '/api/search',
        'search_params': dict(request.args),
        'results': ['result1', 'result2'] if not detection_info['sql_injection_detected'] else [],
        'sql_injection_check': detection_info
    }

    if detection_info['sql_injection_detected']:
        response['error'] = "Invalid search parameters"

    return jsonify(response)


@app.route('/api/login', methods=['POST'])
@sql_injection_middleware
def login_endpoint():
    """
    Login endpoint - common target for authentication bypass.
    Try JSON body: {"username": "admin' OR '1'='1", "password": "anything"}
    """
    detection_info = {
        'sql_injection_detected': getattr(g, 'sql_injection_detected', False),
        'confidence': getattr(g, 'sql_injection_confidence', 0)
    }

    request_data = request.get_json() if request.is_json else {}

    if detection_info['sql_injection_detected']:
        return jsonify({
            'success': False,
            'message': 'Authentication failed - suspicious input detected',
            'sql_injection_check': detection_info
        }), 401

    return jsonify({
        'success': True,
        'message': 'Login successful',
        'user': request_data.get('username', 'unknown'),
        'sql_injection_check': detection_info
    })


@app.route('/api/orders', methods=['POST'])
@sql_injection_middleware
def orders_endpoint():
    """
    Orders endpoint - tests form data.
    Try: form-data with order_id = "1'; DROP TABLE orders;--"
    """
    detection_info = {
        'sql_injection_detected': getattr(g, 'sql_injection_detected', False),
        'confidence': getattr(g, 'sql_injection_confidence', 0)
    }

    if detection_info['sql_injection_detected']:
        return jsonify({
            'success': False,
            'message': 'Order creation blocked - SQL injection detected',
            'sql_injection_check': detection_info
        }), 400

    return jsonify({
        'success': True,
        'order_id': 12345,
        'message': 'Order created successfully',
        'sql_injection_check': detection_info
    })


# ============================================
# ADMIN AND MONITORING ENDPOINTS
# ============================================

@app.route('/admin/logs', methods=['GET'])
def get_logs():
    """Get SQL injection attempt logs"""
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)

    logs = logger.attempts[-limit-offset:][-limit:] if logger.attempts else []

    return jsonify({
        'total_count': len(logger.attempts),
        'returned_count': len(logs),
        'logs': logs
    })


@app.route('/admin/stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    stats = logger.get_stats()
    return jsonify(stats)


@app.route('/admin/config', methods=['GET', 'POST'])
def config_endpoint():
    """View or update configuration"""
    if request.method == 'POST':
        data = request.get_json()
        if 'block_malicious' in data:
            Config.BLOCK_MALICIOUS_REQUESTS = data['block_malicious']
        if 'confidence_threshold' in data:
            Config.CONFIDENCE_THRESHOLD = data['confidence_threshold']
            predictor.threshold = data['confidence_threshold']
        return jsonify({'message': 'Configuration updated', 'config': {
            'block_malicious': Config.BLOCK_MALICIOUS_REQUESTS,
            'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
            'debug_mode': Config.DEBUG_MODE
        }})

    return jsonify({
        'block_malicious': Config.BLOCK_MALICIOUS_REQUESTS,
        'confidence_threshold': Config.CONFIDENCE_THRESHOLD,
        'debug_mode': Config.DEBUG_MODE,
        'monitored_routes': Config.MONITORED_ROUTES,
        'excluded_routes': Config.EXCLUDED_ROUTES
    })


@app.route('/detect_single', methods=['POST'])
def detect_single():
    """
    Standalone detection endpoint - test individual SQL queries.
    Useful for testing the detection model directly.
    """
    data = request.get_json()
    sql_query = data.get('sql_query', '')

    if not sql_query:
        return jsonify({'error': 'No SQL query provided'}), 400

    prediction = predictor.predict(sql_query)

    return jsonify({
        'sql_query': sql_query[:200] + ('...' if len(sql_query) > 200 else ''),
        'prediction': prediction
    })


@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """
    Batch detection endpoint - test multiple queries at once.
    """
    data = request.get_json()
    queries = data.get('queries', [])

    if not queries:
        return jsonify({'error': 'No queries provided'}), 400

    results = []
    for query in queries:
        prediction = predictor.predict(query)
        results.append({
            'query_preview': query[:100] + ('...' if len(query) > 100 else ''),
            'is_malicious': prediction['is_malicious'],
            'confidence': prediction['confidence'],
            'attack_type': prediction.get('attack_type')
        })

    return jsonify({
        'total': len(queries),
        'malicious_count': sum(1 for r in results if r['is_malicious']),
        'results': results
    })


# ============================================
# MAIN ENTRY POINT
# ============================================

def print_test_instructions():
    """Print instructions for testing with Postman"""
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║     SQL INJECTION DETECTION MIDDLEWARE - POSTMAN TESTING         ║
    ╚══════════════════════════════════════════════════════════════════╝

    📍 SERVER URL: http://localhost:5000

    🔬 TEST ENDPOINTS:

    1️⃣  BENIGN REQUESTS (Should NOT trigger detection):
    ───────────────────────────────────────────────────────────────
    GET  /api/users?user_id=123&name=john
    GET  /api/products/123
    POST /api/login {"username": "admin", "password": "secret"}
    POST /api/search {"query": "laptop"}

    2️⃣  MALICIOUS REQUESTS (Should trigger detection):
    ───────────────────────────────────────────────────────────────
    # Union-based injection
    GET  /api/users?id=1 UNION SELECT * FROM users

    # Boolean-based injection
    GET  /api/users?id=1' OR '1'='1

    # Time-based injection
    POST /api/search {"query": "test' AND SLEEP(5)--"}

    # Stacked queries
    POST /api/orders {"order_id": "1; DROP TABLE orders;--"}

    # Authentication bypass
    POST /api/login {"username": "admin'--", "password": "anything"}

    # Path parameter injection
    GET  /api/products/1' OR '1'='1

    # Comment injection
    GET  /api/users?id=1'-- -

    3️⃣  MONITORING ENDPOINTS:
    ───────────────────────────────────────────────────────────────
    GET  /health          - Check API health and stats
    GET  /admin/stats     - View detection statistics
    GET  /admin/logs      - View all detection logs
    POST /detect_single   - Test single SQL query

    4️⃣  TOGGLE BLOCKING MODE:
    ───────────────────────────────────────────────────────────────
    POST /admin/config {"block_malicious": true}
    POST /admin/config {"block_malicious": false}

    📊 EXPECTED BEHAVIOR:
    - Benign requests: Status 200, is_malicious = false
    - Malicious requests: Status 200 (if blocking disabled) or 403 (if blocking enabled)
    - All detection logs are saved to sql_injection_logs.json

    💡 TIP: Check the response 'sql_injection_check' field to see detection results!
    """
    print(instructions)


if __name__ == '__main__':
    print_test_instructions()

    if not predictor.model_loaded:
        print("\n⚠ WARNING: No ML model loaded. Using rule-based detection.")
        print("  For best results, train the model and provide the path.\n")

    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG_MODE
    )
