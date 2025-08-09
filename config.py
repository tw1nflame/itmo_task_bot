# Конфигурация проекта
import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# API ключи
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Настройки для DeepSeek
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

# Настройки эмбеддингов и RAG
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', "paraphrase-multilingual-mpnet-base-v2")
RETRIEVER_K = int(os.getenv('RETRIEVER_K', '5'))
VECTORSTORE_DIR = os.getenv('VECTORSTORE_DIR', 'faiss_index')

# Файлы данных (с значениями по умолчанию)
PROGRAMS_JSON_FILE = os.getenv('JSON_DATA_FILE', 'itmo_programs_full.json')

# Логирование
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

