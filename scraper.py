import requests
import json
import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from datetime import datetime
from pprint import pprint  # Для красивого вывода словарей

# --- Вспомогательные функции ---

# Словарь для перевода месяцев на русский
RU_MONTHS = {
    1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля',
    5: 'мая', 6: 'июня', 7: 'июля', 8: 'августа',
    9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'
}

def format_date_ru(iso_date_str):
    """Преобразует дату из формата ISO в читаемый русский формат."""
    try:
        dt_object = datetime.fromisoformat(iso_date_str)
        return f"{dt_object.day} {RU_MONTHS[dt_object.month]} {dt_object.year} г."
    except (ValueError, KeyError):
        return iso_date_str # Возвращаем как есть, если формат другой

def parse_academic_plan_pdf(pdf_url):
    """
    Скачивает и парсит PDF-файл с учебным планом.
    Переписанная версия с правильным пониманием структуры PDF.
    """
    if not pdf_url:
        print("URL для PDF не предоставлен.")
        return []

    try:
        print(f"Скачиваю PDF: {pdf_url}")
        pdf_response = requests.get(pdf_url)
        pdf_response.raise_for_status()
        pdf_document = fitz.open(stream=pdf_response.content, filetype="pdf")
        print(f"PDF успешно открыт. Количество страниц: {len(pdf_document)}")
    except Exception as e:
        print(f"Не удалось скачать или открыть PDF: {e}")
        return []

    full_text = ""
    for page_num, page in enumerate(pdf_document):
        page_text = page.get_text()
        full_text += page_text

    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    
    # Сохраняем для отладки
    with open('pdf_debug.txt', 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            f.write(f"{i:3d}: {line}\n")
    
    courses = []
    current_block = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Определяем заголовок блока (не число, содержит "дисциплины" или "Пул")
        if not line.isdigit() and (
            re.search(r"дисциплины", line, re.IGNORECASE) or re.search(r"Пул", line, re.IGNORECASE)
        ):
            current_block = line
            # После заголовка всегда две строки с числами (например, 12, 432)
            i += 3
            continue

        # Дисциплина начинается с числа (номер семестра), затем название, затем два числа
        if i + 3 < len(lines):
            if (
                lines[i].isdigit() and
                not lines[i+1].isdigit() and
                lines[i+2].isdigit() and
                lines[i+3].isdigit()
            ):
                semester = int(lines[i])
                discipline_name = lines[i+1].strip()
                credits = int(lines[i+2])
                hours = int(lines[i+3])
                courses.append({
                    "title": discipline_name,
                    "semester": semester,
                    "block": current_block,
                    "credits": credits,
                    "hours": hours
                })
                print(f"Найдена дисциплина: {discipline_name} (семестр {semester}, {credits} кредитов, блок: {current_block})")
                i += 4
                continue

        i += 1

    # Пост-обработка: удаляем дубликаты и фильтруем
    filtered_courses = []
    seen_titles = set()
    for course in courses:
        title = course["title"].strip()
        if (len(title) > 5 and 
            title not in seen_titles and
            not re.match(r"^(Семестры|Наименование|Трудоемкость)", title, re.IGNORECASE)):
            seen_titles.add(title)
            filtered_courses.append(course)

    print(f"\nСтатистика парсинга:")
    print(f"Найдено дисциплин (до фильтрации): {len(courses)}")
    print(f"Найдено дисциплин (после фильтрации): {len(filtered_courses)}")
    pdf_document.close()
    return filtered_courses

def parse_itmo_program(url):
    """
    Основная функция, которая парсит страницу и вызывает парсер для PDF.
    """
    print(f"--- Парсинг данных с {url} ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к сайту: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    next_data_script = soup.find('script', {'id': '__NEXT_DATA__'})

    if not next_data_script:
        print("Ошибка: Тег __NEXT_DATA__ не найден на странице.")
        return None

    data = json.loads(next_data_script.string)
    # Для отладки можно раскомментировать:
    # print(json.dumps(data, indent=2, ensure_ascii=False))
    
    page_props = data.get('props', {}).get('pageProps', {})
    
    # --- Извлечение данных ---
    api_program = page_props.get('apiProgram', {})
    json_program = page_props.get('jsonProgram', {})
    supervisor_info = page_props.get('supervisor', {})

    # Контакты менеджера
    manager_name = f"{supervisor_info.get('firstName', '')} {supervisor_info.get('lastName', '')}".strip()
    manager_contacts = json_program.get('supervisor', {})

    # Даты экзаменов
    exam_dates_raw = page_props.get('examDates', [])
    exam_dates_formatted = [format_date_ru(date) for date in exam_dates_raw]

    # Ссылка на PDF с учебным планом
    academic_plan_url = api_program.get('academic_plan')

    # Название программы
    program_name = api_program.get('title')
    print(f"Название программы: {program_name}")

    # Стоимость обучения
    cost = api_program.get('educationCost', {}).get('russian')
    print(f"Стоимость (РФ): {cost} ₽")

    # Описание программы
    description = json_program.get('about', {}).get('desc', '')
    clean_description = BeautifulSoup(description, 'html.parser').get_text(separator=' ', strip=True)
    print(f"\nОписание: {clean_description}")

    # Часто задаваемые вопросы (FAQ)
    faq_list = json_program.get('faq', [])
    print("\n--- FAQ ---")
    for item in faq_list:
        question = item.get('question')
        answer_html = item.get('answer')
        answer_text = BeautifulSoup(answer_html, 'html.parser').get_text(separator=' ', strip=True)
        print(f"В: {question}\nО: {answer_text}\n")

    # Ссылка на PDF с учебным планом
    print(f"Ссылка на учебный план (PDF): {academic_plan_url}")

    # --- Вызываем парсер для PDF ---
    print(f"Начинаю парсинг учебного плана из PDF: {academic_plan_url}")
    curriculum = parse_academic_plan_pdf(academic_plan_url)
    print(f"Найдено {len(curriculum)} дисциплин в учебном плане.")

    # Собираем все в один словарь
    program_info = {
        'url': url,
        'program_name': program_name,
        'manager': {
            'name': manager_name,
            'email': manager_contacts.get('email'),
            'phone': manager_contacts.get('phone')
        },
        'admission_info': {
            'cost_rub': cost,
            'cost_period': 'в год',
            'military_department': "Да" if api_program.get('isMilitary') else "Нет",
            'dormitory': "Да", # На странице указано "Да", можно сделать более умный парсинг, но для задачи этого достаточно
            'exam_dates': exam_dates_formatted,
            'academic_plan_pdf': academic_plan_url
        },
        'about': clean_description,
        'faq': [
            {
                'question': item.get('question'),
                'answer': BeautifulSoup(item.get('answer', ''), 'html.parser').get_text(separator=' ', strip=True)
            } for item in faq_list
        ],
        'curriculum': curriculum # <<<--- ДОБАВИЛИ УЧЕБНЫЙ ПЛАН
    }
    
    return program_info

# --- Основной блок для запуска ---
if __name__ == "__main__":
    urls_to_parse = [
        "https://abit.itmo.ru/program/master/ai",
        "https://abit.itmo.ru/program/master/ai_product"
    ]
    
    all_programs_data = []
    for program_url in urls_to_parse:
        parsed_data = parse_itmo_program(program_url)
        if parsed_data:
            all_programs_data.append(parsed_data)
            print("Данные по программе успешно извлечены.")
            # pprint(parsed_data, indent=2, width=120) # Можно раскомментировать для детального просмотра
            print("\n" + "="*80 + "\n")

    # Теперь all_programs_data содержит структурированную информацию
    # по обеим программам, готовую для сохранения в JSON и использования в боте.
    
    # Сохранение в JSON-файл
    if all_programs_data:
        with open('itmo_programs_full.json', 'w', encoding='utf-8') as f:
            json.dump(all_programs_data, f, ensure_ascii=False, indent=4)
        print("Все данные, включая учебные планы, сохранены в файл 'itmo_programs_full.json'")