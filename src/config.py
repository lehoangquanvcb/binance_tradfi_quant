from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
OUTPUTS = ROOT / 'outputs'

for p in [DATA_RAW, DATA_PROCESSED, MODELS, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / '.env')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
TRADING_MODE = os.getenv('TRADING_MODE', 'paper').lower()
