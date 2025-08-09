import os
import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from data_processor import prepare_documents_from_json
from config import EMBEDDINGS_MODEL, PROGRAMS_JSON_FILE, VECTORSTORE_DIR


def main():
    print("Подготовка документов...")
    docs = prepare_documents_from_json(PROGRAMS_JSON_FILE)
    if not docs:
        raise RuntimeError("Документы не найдены для индексации")

    print(f"Создание эмбеддингов: {EMBEDDINGS_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    print("Построение FAISS индекса...")
    vs = FAISS.from_documents(docs, embeddings)

    # Обновляем директорию индекса
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    print(f"Сохранение индекса в: {VECTORSTORE_DIR}")
    vs.save_local(VECTORSTORE_DIR)
    print("Готово.")


if __name__ == "__main__":
    main()
