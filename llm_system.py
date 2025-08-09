from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import (
    EMBEDDINGS_MODEL, RETRIEVER_K,
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
    VECTORSTORE_DIR
)

class RAGSystem:
    """Система вопросов и ответов на основе RAG (Retrieval-Augmented Generation)."""
    
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings_model = None
        self.program_names = []  # Список доступных программ
        self.conversation_history = {}  # История диалогов по user_id
    
    def initialize(self, documents: List[Document] | None) -> None:
        """Инициализирует RAG систему: загружает сохраненный индекс или строит новый.

        Если локальный индекс существует, загружает его. Иначе, если переданы документы,
        строит индекс и сохраняет. Если ни индекса, ни документов нет — ошибка.
        """
        print("Загрузка модели эмбеддингов (может занять время)...")
        try:
            self.embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели эмбеддингов: {e}")
        
        # Пытаемся загрузить сохраненный индекс
        loaded_from_disk = False
        try:
            import os
            if os.path.exists(VECTORSTORE_DIR):
                print(f"Найден сохраненный индекс. Загрузка из: {VECTORSTORE_DIR}")
                self.vectorstore = FAISS.load_local(
                    VECTORSTORE_DIR,
                    self.embeddings_model,
                    allow_dangerous_deserialization=True
                )
                loaded_from_disk = True
                # Извлекаем список программ из документов индекса
                self._extract_program_names_from_store()
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить индекс: {e}")

        if not loaded_from_disk:
            if not documents:
                raise RuntimeError(
                    "Локальный индекс не найден и документы не предоставлены. "
                    "Запустите скрипт сборки индекса или передайте документы."
                )

            # Извлекаем список программ из метаданных документов
            self.program_names = list(set(doc.metadata.get('source', '') for doc in documents if doc.metadata.get('source')))
            print(f"Обнаружено программ: {len(self.program_names)}")
            for prog in self.program_names:
                print(f"  - {prog}")

            print("Создание векторной базы FAISS...")
            try:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings_model)
            except Exception as e:
                raise RuntimeError(f"Ошибка создания векторной базы: {e}")

            # Сохраняем индекс на диск для повторного использования
            try:
                print(f"Сохранение индекса в: {VECTORSTORE_DIR}")
                self.vectorstore.save_local(VECTORSTORE_DIR)
            except Exception as e:
                print(f"Предупреждение: не удалось сохранить индекс: {e}")

        print("Создание RAG-цепочки...")
        self.qa_chain = self._create_rag_chain()
        print("RAG система готова!")

    def _extract_program_names_from_store(self):
        """Заполняет список program_names из документов векторного хранилища."""
        try:
            all_docs = list(self.vectorstore.docstore._dict.values())
            self.program_names = list(set(doc.metadata.get('source', '') for doc in all_docs if doc.metadata.get('source')))
            print(f"Обнаружено программ (из индекса): {len(self.program_names)}")
            for prog in self.program_names:
                print(f"  - {prog}")
        except Exception as e:
            print(f"Предупреждение: не удалось извлечь список программ из индекса: {e}")
    
    def _create_rag_chain(self):
        """Создает и возвращает RAG-цепочку LangChain."""
        if not self.vectorstore:
            raise RuntimeError("Векторная база не инициализирована")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})
        
        # Используем DeepSeek как провайдер LLM
        try:
            if not DEEPSEEK_API_KEY:
                raise RuntimeError("DEEPSEEK_API_KEY не найден в переменных окружения")
            
            llm = ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=0
            )
            print(f"Используется DeepSeek модель: {DEEPSEEK_MODEL}")
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации DeepSeek: {e}")

        template = """
        Ты — консультант по программам магистратуры ИТМО. Отвечай естественно и профессионально простым текстом без смайликов и без markdown форматирования.
        
        ПРАВИЛА:
        1. Используй ТОЛЬКО предоставленную информацию
        2. Если спрашивают о программах в общем, упоминай ВСЕ доступные программы
        3. Если спрашивают о конкретной программе, отвечай ТОЛЬКО о ней
        4. Не путай разные программы между собой
        5. Четко разделяй информацию по программам
        6. Если нет информации, скажи: "К сожалению, у меня нет точной информации по этому вопросу"

        ИНФОРМАЦИЯ:
        {context}

        ВОПРОС: {question}

        ОТВЕТ:
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    
    def _add_to_history(self, user_id: str, user_message: str, bot_response: str):
        """Добавляет пару сообщений в историю."""
        if not user_id:
            return
            
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': 'recent'
        })
        
        # Ограничиваем историю последними 3 парами
        if len(self.conversation_history[user_id]) > 3:
            self.conversation_history[user_id] = self.conversation_history[user_id][-3:]
    
    def _get_recent_context(self, user_id: str) -> str:
        """Получает контекст последнего обмена сообщениями."""
        if not user_id or user_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[user_id]
        if not history:
            return ""
        
        # Берем только последний обмен
        last_exchange = history[-1]
        
        # Проверяем, был ли последний ответ бота запросом уточнения
        bot_response = last_exchange['bot'].lower()
        clarification_indicators = [
            'по какой программе', 'уточнить программу', 'выбрать программу',
            'какую программу', 'программе вас интересует'
        ]
        
        if any(indicator in bot_response for indicator in clarification_indicators):
            return f"КОНТЕКСТ: Ранее пользователь спрашивал: '{last_exchange['user']}', и бот попросил уточнить программу."
        
        return ""
    
    def _classify_question(self, question: str, user_id: str = None) -> dict:
        """Классифицирует вопрос и извлекает программу если нужно."""
        try:
            llm = ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=0
            )
            
            available_programs = '\n'.join([f"- {prog}" for prog in self.program_names])
            
            # Получаем контекст предыдущего обмена
            context = self._get_recent_context(user_id) if user_id else ""
            question_with_context = f"{context}\n\nТЕКУЩИЙ ВОПРОС: {question}" if context else question
            
            template = """
            Проанализируй вопрос пользователя о программах магистратуры ИТМО.
            
            ДОСТУПНЫЕ ПРОГРАММЫ:
            {programs}
            
            ВОПРОС: {question}
            
            ПРАВИЛА КЛАССИФИКАЦИИ:
            - type = "recommendation" если:
              * Описывают свой опыт/навыки и просят совет о программах или предметах
              * Просят рекомендации на основе background ("мой опыт такой-то, что подходит?")
              * Спрашивают какие предметы/курсы выбрать с описанием опыта
              * Просят помочь с выбором на основе их профиля
              * Есть фразы "я работаю", "мой опыт", "посоветуешь" с контекстом опыта
              
            ОПРЕДЕЛЕНИЕ ПРОГРАММЫ для recommendation:
            - program = "название программы" ТОЛЬКО если в вопросе есть "на программе X", "по программе X", "в программе X"
            - program = null если спрашивают "какую программу выбрать" или просто "что посоветуешь"
            
            - type = "clarification_needed" если:
              * Просят рекомендации об элективах/предметах/курсах с описанием опыта НО не указывают программу
              * Спрашивают "что выбрать из элективов" с опытом, но неясно по какой программе
              * Нужно уточнить программу для персональной рекомендации
            
            - type = "program_followup" если:
              * В контексте видно, что бот недавно просил уточнить программу
              * Пользователь отвечает названием программы или коротким ответом
              * Это ответ на запрос уточнения программы
              
            - type = "general" если:
              * Спрашивают о списке программ без описания опыта
              * Общие вопросы о поступлении, стоимости
              * СРАВНИТЕЛЬНЫЕ вопросы ("какая программа лучше", "на какой программе больше", "чем отличаются программы")
              * Вопросы типа "какой из программ", "где больше", "что лучше выбрать" БЕЗ описания опыта
              
            - type = "specific" если:
              * Задают общие вопросы о КОНКРЕТНО НАЗВАННОЙ программе без описания опыта
              * Спрашивают о стоимости, требованиях, датах по конкретной программе  
              * НО НЕ сравнительные вопросы между программами
            
            Ответь ТОЛЬКО валидным JSON:
            {{"type": "recommendation", "program": null}} или {{"type": "general", "program": null}} или {{"type": "specific", "program": "название программы"}} или {{"type": "clarification_needed", "program": null}} или {{"type": "program_followup", "program": "название программы"}}
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "question": question_with_context,
                "programs": available_programs
            }).strip()
            
            # Парсим JSON ответ
            import json
            try:
                # Убираем возможные markdown блоки кода
                response_clean = response.strip()
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:]  # убираем ```json
                if response_clean.endswith('```'):
                    response_clean = response_clean[:-3]  # убираем ```
                response_clean = response_clean.strip()
                
                result = json.loads(response_clean)
                print(f"DEBUG: LLM классификация успешна: {result}")
                return {
                    'type': result.get('type', 'general'),
                    'program': result.get('program')
                }
            except json.JSONDecodeError:
                print(f"DEBUG: Не удалось распарсить ответ LLM: {response}")
                # По умолчанию считаем общим вопросом
                return {'type': 'general', 'program': None}
            
        except Exception as e:
            print(f"DEBUG: Ошибка классификации вопроса: {e}")
            # По умолчанию считаем общим вопросом
            return {'type': 'general', 'program': None}
    
    def ask(self, question: str, user_id: str = None) -> str:
        """Задает вопрос RAG системе и возвращает ответ."""
        if not self.qa_chain:
            return "Система не инициализирована. Обратитесь к администратору."
        
        try:
            # Первый вызов LLM - классификация и извлечение программы
            classification = self._classify_question(question, user_id)
            question_type = classification['type']
            target_program = classification['program']
            
            print(f"DEBUG: Тип вопроса: {question_type.upper()}")
            if target_program:
                print(f"DEBUG: Извлеченная программа: {target_program}")
            
            # Специальная обработка ответа на уточнение программы
            if question_type == 'program_followup' and target_program and user_id:
                print(f"DEBUG: Обрабатываем ответ на уточнение программы: {target_program}")
                
                # Восстанавливаем исходный вопрос из истории
                if user_id in self.conversation_history and self.conversation_history[user_id]:
                    original_question = self.conversation_history[user_id][-1]['user']
                    print(f"DEBUG: Исходный вопрос: {original_question}")
                    
                    # Формируем новый вопрос с указанием программы
                    enhanced_question = f"{original_question} на программе {target_program}"
                    
                    # Перенаправляем как рекомендацию для конкретной программы
                    return self._handle_recommendation_with_program(enhanced_question, target_program, user_id, original_question)
            
            if question_type == 'specific' and target_program:
                # Конкретный вопрос о программе - фильтруем поиск по метаданным
                print(f"DEBUG: Ищем документы только по программе: {target_program}")
                
                # Получаем все документы
                all_docs = list(self.vectorstore.docstore._dict.values())
                
                # Фильтруем по программе
                program_docs = [doc for doc in all_docs if doc.metadata.get('source') == target_program]
                print(f"DEBUG: Найдено документов по программе: {len(program_docs)}")
                
                if not program_docs:
                    return f"Информация о программе '{target_program}' не найдена."
                
                # Создаем временную векторную базу только с документами этой программы
                temp_vectorstore = FAISS.from_documents(program_docs, self.embeddings_model)
                
                # Ищем наиболее релевантные документы в рамках программы
                relevant_docs = temp_vectorstore.similarity_search(question, k=RETRIEVER_K)
                
                # Создаем контекст
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                template = f"""
                Ты — консультант по программам магистратуры ИТМО. Отвечай естественно и профессионально простым текстом без смайликов и без markdown форматирования.
                
                ВАЖНО: Вся информация относится к программе "{target_program}".
                
                ИНФОРМАЦИЯ:
                {context}

                ВОПРОС: {question}

                ОТВЕТ:
                """
                
            elif question_type == 'clarification_needed':
                # LLM решила, что нужно уточнение программы для персональной рекомендации
                print("DEBUG: Требуется уточнение программы для рекомендации")
                
                # Генерируем персонализированный ответ-уточнение с помощью LLM
                clarification_template = f"""
                Ты — консультант по программам магистратуры ИТМО. Пользователь задал вопрос, но для персональной рекомендации нужно уточнить программу.
                
                ДОСТУПНЫЕ ПРОГРАММЫ:
                1. "Искусственный интеллект" - фокус на алгоритмах машинного обучения, глубоком обучении, компьютерном зрении
                2. "Управление ИИ-продуктами/AI Product" - фокус на продуктовом менеджменте, бизнес-аналитике, управлении AI-проектами
                
                ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}
                
                ЗАДАЧА: Объясни пользователю, что для персональной рекомендации нужно уточнить программу. Учти контекст его вопроса и опыт. Предложи выбрать программу.
                
                Отвечай простым текстом без смайликов и markdown.
                
                ОТВЕТ:
                """
                
                llm = ChatOpenAI(
                    model=DEEPSEEK_MODEL,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    temperature=0
                )
                
                prompt = ChatPromptTemplate.from_template(clarification_template)
                chain = prompt | llm | StrOutputParser()
                
                response = chain.invoke({"question": question})
                
                # Сохраняем в историю
                if user_id:
                    self._add_to_history(user_id, question, response.strip())
                
                return response.strip()
                
            elif question_type == 'recommendation':
                # Если есть конкретная программа — используем хелпер для единообразия
                if target_program:
                    print(f"DEBUG: Рекомендация для конкретной программы через helper: {target_program}")
                    # enhanced_question = question + ' на программе {target_program}' — не добавляем, чтобы не дублировать, просто передаем question
                    return self._handle_recommendation_with_program(question, target_program, user_id, question)
                else:
                    # Если программа не указана - ищем по всем программам (старый код)
                    print("DEBUG: Рекомендации для всех программ")
                    docs = []
                    all_docs = list(self.vectorstore.docstore._dict.values())
                    recommendation_keywords = ["дисциплин", "курс", "изучается", "практик", "проект", "навык", "технолог", "програм"]
                    combined_query = question + " " + " ".join(recommendation_keywords)
                    for program_name in self.program_names:
                        program_docs = [doc for doc in all_docs if doc.metadata.get('source') == program_name]
                        if program_docs:
                            temp_vectorstore = FAISS.from_documents(program_docs, self.embeddings_model)
                            relevant_program_docs = temp_vectorstore.similarity_search(combined_query, k=5)
                            docs.extend(relevant_program_docs)
                            print(f"DEBUG: Добавлено {len(relevant_program_docs)} документов от программы '{program_name}' для рекомендации")
                    print(f"DEBUG: Контекст для рекомендаций создан из {len(docs)} документов")
                    final_programs = set(doc.metadata.get('source', '') for doc in docs if doc.metadata.get('source'))
                    print(f"DEBUG: Программы в контексте рекомендаций: {', '.join(final_programs)}")
                    context = "\n\n".join([doc.page_content for doc in docs])
                    template = f"""
                    Ты — консультант по программам магистратуры ИТМО. Дай персональные рекомендации на основе опыта пользователя.
                    
                    Отвечай простым текстом без смайликов и markdown.
                    
                    Задача: Проанализируй опыт пользователя и порекомендуй подходящие программы или предметы.
                    
                    Информация о программах и курсах:
                    {context}

                    Вопрос: {question}

                    Ответ:
                    """
                
            else:
                # Общий вопрос - используем полный поиск
                print("DEBUG: Общий вопрос - используем полный поиск")
                
                # Для общих вопросов собираем документы от всех программ
                docs = []
                all_docs = list(self.vectorstore.docstore._dict.values())
                
                # Добавляем по несколько лучших документов от каждой программы
                for program_name in self.program_names:
                    program_docs = [doc for doc in all_docs if doc.metadata.get('source') == program_name]
                    
                    if program_docs:
                        # Создаем временную векторную базу для этой программы
                        temp_vectorstore = FAISS.from_documents(program_docs, self.embeddings_model)
                        # Ищем самые релевантные документы для данной программы
                        relevant_program_docs = temp_vectorstore.similarity_search(question, k=3)
                        docs.extend(relevant_program_docs)
                        print(f"DEBUG: Добавлено {len(relevant_program_docs)} документов от программы '{program_name}'")
                
                print(f"DEBUG: Общий контекст создан из {len(docs)} документов")
                
                # Проверяем финальные программы в контексте
                final_programs = set(doc.metadata.get('source', '') for doc in docs if doc.metadata.get('source'))
                print(f"DEBUG: Программы в финальном контексте: {', '.join(final_programs)}")
                
                context = "\n\n".join([doc.page_content for doc in docs])
                
                template = f"""
                Ты — консультант по программам магистратуры ИТМО. Отвечай естественно и профессионально простым текстом без смайликов и без markdown форматирования.
                
                ВАЖНО: 
                1. Используй ТОЛЬКО предоставленную информацию
                2. У ИТМО есть несколько программ магистратуры - упоминай ВСЕ доступные программы
                3. Четко разделяй информацию по разным программам
                4. Отвечай как эксперт-консультант, а не как система
                
                ИНФОРМАЦИЯ:
                {context}

                ВОПРОС: {question}

                ОТВЕТ:
                """
            
            # Второй вызов LLM - генерация ответа
            llm = ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=0
            )
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"question": question})
            
            # Сохраняем в историю
            if user_id:
                self._add_to_history(user_id, question, response.strip())
            
            return response.strip()
            
        except Exception as e:
            return f"Произошла ошибка при обработке вопроса: {str(e)}"
    
    def _handle_recommendation_with_program(self, enhanced_question: str, target_program: str, user_id: str, original_question: str) -> str:
        """Обрабатывает рекомендацию для конкретной программы."""
        print(f"DEBUG: Рекомендации для программы {target_program} на основе вопроса: {enhanced_question}")
        
        # Получаем все документы и фильтруем по программе
        all_docs = list(self.vectorstore.docstore._dict.values())
        program_docs = [doc for doc in all_docs if doc.metadata.get('source') == target_program]
        
        if not program_docs:
            response = f"К сожалению, информация о программе '{target_program}' не найдена."
        else:
            # Создаем временную векторную базу для программы
            temp_vectorstore = FAISS.from_documents(program_docs, self.embeddings_model)
            
            # Ключевые слова для поиска релевантной информации
            recommendation_keywords = ["дисциплин", "курс", "изучается", "практик", "проект", "навык", "технолог"]
            combined_query = enhanced_question + " " + " ".join(recommendation_keywords)
            
            relevant_docs = temp_vectorstore.similarity_search(combined_query, k=8)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            template = f"""
            Ты — консультант по программам магистратуры ИТМО. Дай персональные рекомендации на основе опыта пользователя.
            
            ПРОГРАММА: {target_program}
            
            ИСХОДНЫЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ: {original_question}
            
            Отвечай простым текстом без смайликов и markdown.
            
            Информация о курсах программы:
            {context}

            Ответ:
            """
            
            llm = ChatOpenAI(
                model=DEEPSEEK_MODEL,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                temperature=0
            )
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"question": enhanced_question})
        
        # Сохраняем в историю
        if user_id:
            self._add_to_history(user_id, f"Программа: {target_program}", response.strip())
        
        return response.strip()
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирована ли система."""
        return self.qa_chain is not None
    
    def get_available_programs(self) -> List[str]:
        """Возвращает список доступных программ."""
        return self.program_names.copy()
    
    def get_program_info(self, program_name: str) -> str:
        """Возвращает информацию о конкретной программе."""
        if program_name not in self.program_names:
            return f"Программа '{program_name}' не найдена. Доступные программы: {', '.join(self.program_names)}"
        
        return self.ask(f"Расскажи подробно о программе {program_name}")
