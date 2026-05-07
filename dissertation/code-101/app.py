# ============================================
# FLASK APPLICATION RUNNER
# Filename: app.py
# ============================================

"""
This file is intentionally small.

The main Flask application, middleware, routes, and prediction logic live in
`sql_injection_middleware.py`. This runner exists so that the project has a
clear and conventional startup command:

    python app.py

That keeps the API entry point separate from the training script and makes it
easier to explain the application structure in the dissertation.
"""

from sql_injection_middleware import app, predictor, print_test_instructions, Config


if __name__ == '__main__':
    print_test_instructions()

    if not predictor.model_loaded:
        print("\n⚠ WARNING: No trained model bundle was loaded.")
        print("  The middleware will fall back to rule-based detection.")
        print("  Run `python task1.py` first if you want the best saved model in the API.\n")

    app.run(
        host='0.0.0.0',
        port=5100,
        debug=Config.DEBUG_MODE
    )
