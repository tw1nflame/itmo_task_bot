"""
Полный запуск: скрапинг -> подготовка документов -> построение индекса -> запуск Telegram-бота.
Весь пайплайн в одном файле, без лишних проверок и условий.
"""
import asyncio
import json
import os
import logging
from scraper import parse_itmo_program
from data_processor import prepare_documents_from_json
from config import EMBEDDINGS_MODEL, PROGRAMS_JSON_FILE, VECTORSTORE_DIR, LOG_LEVEL, LOG_FORMAT, TELEGRAM_BOT_TOKEN
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from llm_system import RAGSystem
from aiogram import Bot, Dispatcher, types, F


def main():
    # 1. Скрапинг
    urls = [
        "https://abit.itmo.ru/program/master/ai",
        "https://abit.itmo.ru/program/master/ai_product"
    ]
    all_programs_data = []
    for url in urls:
        data = parse_itmo_program(url)
        if data:
            all_programs_data.append(data)
    with open(PROGRAMS_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(all_programs_data, f, ensure_ascii=False, indent=4)
    print("Скрапинг завершён, данные сохранены.")

    # 2. Построение индекса
    docs = prepare_documents_from_json(PROGRAMS_JSON_FILE)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vs = FAISS.from_documents(docs, embeddings)
    if os.path.exists(VECTORSTORE_DIR):
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vs.save_local(VECTORSTORE_DIR)
    print("Индекс построен и сохранён.")

    # 3. Telegram-бот
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL))
    logger = logging.getLogger("main")
    rag_system = RAGSystem()
    rag_system.initialize(documents=None)
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    @dp.message(F.text)
    async def message_handler(message: types.Message):
        user_id = str(message.from_user.id)
        user_message = message.text
        logger.info(f"Сообщение от пользователя {user_id}: {user_message}")
        response = rag_system.ask(user_message, user_id=user_id)
        await message.answer(response, parse_mode='Markdown')
        logger.info(f"Ответ пользователю {user_id}: {response[:100]}...")

    async def main_async():
        logger.info("Запуск Telegram бота на aiogram...")
        try:
            await dp.start_polling(bot)
        finally:
            await bot.session.close()
            logger.info("Бот остановлен")

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
