# passenger_wsgi.py
import os, sys

# Ensure this folder is on sys.path
APP_DIR = os.path.dirname(__file__)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# Optional: keep temp/cache local to the app
os.environ.setdefault("TMPDIR", os.path.join(APP_DIR, "tmp"))
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Import Flask app object from app.py
from app import app as application  # <- app.py defines: app = Flask(__name__)
