# ============================================
# TRAIN AND SAVE THE BEST MODEL
# Filename: train_and_save_model.py
# ============================================

"""
This helper script exists for deployment convenience.

Instead of maintaining a second, simplified training path, it simply calls the
main dissertation training script (`task1.py`). That ensures the model saved
for the middleware is the same model selected from the full evaluation process.
"""

import subprocess
import sys


def train_and_save_model():
    """
    Run the main training script so that the best model bundle is produced for
    the Flask middleware.
    """
    command = [sys.executable, 'task1.py']
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(train_and_save_model())
