import json
from typing import List, Dict, Any
from langchain_core.documents import Document

def prepare_documents_from_json(file_path: str) -> List[Document]:
    """
    Загружает данные из JSON и преобразует их в список
    объектов Document для LangChain.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден. Запустите scraper.py для создания данных.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON файла: {e}")

    documents = []
    for program in data:
        program_name = program.get('program_name', 'Неизвестная программа')
        meta = {"source": program_name}
        
        # Общая информация о программе
        documents.append(Document(
            page_content=f"Программа магистратуры ИТМО '{program_name}': {program.get('about')}", 
            metadata=meta
        ))
        
        admission_info = program.get('admission_info', {})
        if admission_info.get('cost_rub'):
            documents.append(Document(
                page_content=f"Стоимость обучения на программе '{program_name}': {admission_info['cost_rub']} рублей {admission_info.get('cost_period', '')}.", 
                metadata=meta
            ))
        
        documents.append(Document(
            page_content=f"Военная кафедра на программе '{program_name}': {admission_info.get('military_department', 'Информация не указана')}.", 
            metadata=meta
        ))
        
        manager = program.get('manager', {})
        if manager.get('name'):
            documents.append(Document(
                page_content=f"Контакты менеджера программы '{program_name}': {manager.get('name', '')}, email: {manager.get('email', '')}, телефон: {manager.get('phone', '')}.", 
                metadata=meta
            ))
        
        exam_dates = admission_info.get('exam_dates', [])
        if exam_dates:
            exam_dates_str = ", ".join(exam_dates)
            documents.append(Document(
                page_content=f"Даты экзаменов на программу '{program_name}': {exam_dates_str}.", 
                metadata=meta
            ))

        # FAQ
        for faq_item in program.get('faq', []):
            if faq_item.get('question') and faq_item.get('answer'):
                documents.append(Document(
                    page_content=f"Вопрос по программе '{program_name}': {faq_item['question']}. Ответ: {faq_item['answer']}", 
                    metadata=meta
                ))

        # Учебный план
        for course in program.get('curriculum', []):
            if course.get('title'):
                block = course.get('block', 'Блок не указан')
                documents.append(Document(
                    page_content=f"Дисциплина '{course['title']}' изучается на программе магистратуры ИТМО '{program_name}' в {course.get('semester', 'неизвестном')} семестре. Блок: {block}. Тип: {course.get('type', 'неизвестная')} дисциплина. Трудоемкость: {course.get('credits', 0)} з.е.",
                    metadata={
                        "source": program_name, 
                        "type": course.get('type', ''), 
                        "semester": course.get('semester', 0),
                        "block": block
                    }
                ))
    
    return documents

def load_programs_data(file_path: str) -> List[Dict[str, Any]]:
    """Загружает данные программ из JSON файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден. Запустите scraper.py для создания данных.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON файла: {e}")

def get_all_electives(programs_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Извлекает все выборные дисциплины с информацией о программе."""
    all_electives = []
    for program in programs_data:
        program_name = program.get('program_name', 'Неизвестная программа')
        for course in program.get('curriculum', []):
            if course.get('type') == 'Выборная':
                course_copy = course.copy()
                course_copy['program_name'] = program_name
                all_electives.append(course_copy)
    
    return all_electives
