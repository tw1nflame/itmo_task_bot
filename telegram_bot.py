"""
Минималистичный Telegram-бот на aiogram: принимает текст и отвечает через RAGSystem.
"""

import os
import json
import logging
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram import F
from data_processor import prepare_documents_from_json
from llm_system import RAGSystem
from config import TELEGRAM_BOT_TOKEN, PROGRAMS_JSON_FILE, LOG_LEVEL, LOG_FORMAT, VECTORSTORE_DIR

# Настройка логирования
logging.basicConfig(
    format=LOG_FORMAT,
    level=getattr(logging, LOG_LEVEL)
)
logger = logging.getLogger(__name__)

class ITMOChatBot:
    """Минимальный класс чат-бота: только вопрос-ответ через RAG."""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Инициализирует чат-бота."""
        try:
            logger.info("Инициализация чат-бота ИТМО...")
            
            # Проверяем наличие файла с данными
            if not os.path.exists(PROGRAMS_JSON_FILE):
                logger.error(f"Файл с данными не найден: {PROGRAMS_JSON_FILE}")
                return False
            
            # Если индекс уже сохранен, загружаем его без пересборки, иначе строим
            if os.path.exists(VECTORSTORE_DIR):
                logger.info(f"Обнаружен сохраненный индекс: {VECTORSTORE_DIR}. Загрузка без пересборки.")
                self.rag_system.initialize(documents=None)
            else:
                # Подготавливаем документы для RAG только если нужно строить индекс
                logger.info("Подготовка документов для RAG системы (индекс не найден)...")
                documents = prepare_documents_from_json(PROGRAMS_JSON_FILE)
                
                if not documents:
                    logger.error("Не удалось подготовить документы")
                    return False
                
                # Инициализируем RAG систему и сохраним индекс
                self.rag_system.initialize(documents)
            
            self.is_initialized = True
            logger.info("Чат-бот успешно инициализирован!")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации чат-бота: {e}")
            return False
    
    def process_message(self, user_id: str, message: str) -> str:
        """Обрабатывает входящее сообщение: передает его в RAG и возвращает ответ."""
        if not self.is_initialized:
            return "Чат-бот не инициализирован. Обратитесь к администратору."
        
        try:
            # Простой режим: любой текст идет в RAG с учетом user_id (для истории)
            return self.rag_system.ask(message, user_id=user_id)
            
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            return "Произошла ошибка при обработке вашего сообщения. Попробуйте еще раз."

# Создаем глобальный экземпляр чат-бота
chatbot = ITMOChatBot()

# Создаем бота и диспетчер
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Команд и меню нет — только обработчик произвольного текста

@dp.message(F.text)
async def message_handler(message: types.Message):
    """Обработчик обычных текстовых сообщений."""
    try:
        user_id = str(message.from_user.id)
        user_message = message.text
        
        # Логируем сообщение пользователя
        logger.info(f"Сообщение от пользователя {user_id}: {user_message}")
        
        # Обрабатываем сообщение через чат-бота
        response = chatbot.process_message(user_id, user_message)
        
        # Отправляем ответ
        await message.answer(response, parse_mode='Markdown')
        
        # Логируем ответ
        logger.info(f"Ответ пользователю {user_id}: {response[:100]}...")
        
    except Exception as e:
        logger.error(f"Ошибка обработки сообщения: {e}")
        await message.answer(
            "Произошла ошибка при обработке вашего сообщения. Попробуйте еще раз."
        )

async def main():
    """Основная функция для запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("Не указан TELEGRAM_BOT_TOKEN")
        return
    
    # Инициализируем чат-бота
    if not chatbot.initialize():
        logger.error("Не удалось инициализировать чат-бота")
        return
    
    logger.info("Запуск Telegram бота на aiogram...")
    
    try:
        # Запускаем бота
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания")
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
    finally:
        await bot.session.close()
        logger.info("Бот остановлен")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nБот остановлен пользователем")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
