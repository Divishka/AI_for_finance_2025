import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import os
from openai import OpenAI, APIConnectionError, RateLimitError
import pickle
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import InMemoryVectorStore

# 1. Загрузка данных
train_data = pd.read_csv('./data/train_data.csv')

# 2. Очистка текста от лишних символов
def clean_text(text):
    # Удаляем символы Markdown (#, ** и др.) и прочие нетекстовые элементы
    text = re.sub(r'[#*]+', '', text)  # Удаляем # и *
    text = re.sub(r'\[.*?\]', '', text)  # Удаляем содержимое в квадратных скобках
    text = re.sub(r'\s+', ' ', text)  # Нормализуем пробелы
    text = text.strip()  # Удаляем лишние пробелы по краям
    return text

# Применяем очистку к столбцу 'text'
train_data['cleaned_text'] = train_data['text'].apply(clean_text)

# 2. Токенизатор и разбивка на чанки
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit(1)
# Функция для подсчёта токенов
def count_tokens(text):
    return len(tokenizer(text, truncation=False, padding=False)['input_ids'])

# 4. Разбивка текста на эмбеддинги (чанки)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Целевая длина в токенах
    chunk_overlap=50,  # Перекрытие в токенах
    length_function=count_tokens,  # Функция подсчёта длины
)

# Разбиваем текст для каждой строки
chunks_list = []
for text in train_data['cleaned_text']:
    chunks = text_splitter.split_text(text)
    chunks_list.append(chunks)

# Добавляем разбитые тексты обратно в DataFrame
train_data['chunks'] = chunks_list

# 4. Подготовка документов
documents = []
for _, row in train_data.iterrows():
    for chunk in row['chunks']:
        metadata={"id": row['id'], 
                      "source": "train_data", 
                      "tags": row.get('tags', []), 
                      "annotation": row.get('annotation', "")}
        doc = Document(
            page_content=chunk, 
            metadata=metadata
            )
        documents.append(doc)

# 6. Формирование текстов для эмбеддинга
texts = [doc.page_content for doc in documents]
MAX_TEXTS = 10000
texts = texts[:MAX_TEXTS]
print(f"Обрабатываем {len(texts)} текстов (лимит: {MAX_TEXTS})")

if not texts:
    print("Список texts пуст. Завершаем работу.")
    exit()

# 7. Функция для генерации эмбеддингов
client = OpenAI(
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key="sk-k4GzLvBEsBYNbtVPpDaEMg"
)

def get_openai_embeddings(texts, model="text-embedding-3-small"):
    """Генерирует эмбеддинги через OpenAI API"""
    embeddings = []
    for i, text in enumerate(texts):
        if i % 10 == 0:  # Каждые 100 документов
            print(f"Обработка {i}/{len(texts)}...")
        try:
            response = client.embeddings.create(model=model, input=text, timeout=30)
            embeddings.append(response.data[0].embedding)
        except APIConnectionError as e:
            print(f"Ошибка подключения для текста {i}: {e}")
            embeddings.append([0.0] * 1536)
        except RateLimitError as e:
            print(f"Превышен лимит для текста {i}: {e}")
            embeddings.append([0.0] * 1536)
        except Exception as e:
            print(f"Неизвестная ошибка для текста {i}: {e}")
            embeddings.append([0.0] * 1536)
    return embeddings


BATCH_SIZE = 100
EMBEDDINGS_CACHE = "./embeddings_cache.pkl"

# 1. Загрузка кеша или начало с нуля
if os.path.exists(EMBEDDINGS_CACHE):
    print("Загружаем кешированные эмбеддинги...")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        embeddings_array = pickle.load(f)
    # С какого индекса начинать следующую порцию
    start_idx = len(embeddings_array)
    print(f"Продолжаем с индекса {start_idx}")
else:
    embeddings_array = []
    start_idx = 0
    print(f"Начинаем генерацию эмбеддингов для {len(texts)} текстов...")

# 2. Обработка порциями
for i in range(start_idx, len(texts), BATCH_SIZE):
    # Берём порцию: от i до i + BATCH_SIZE (или до конца)
    batch = texts[i:i + BATCH_SIZE]
    print(f"Обработка порции {i}–{min(i + len(batch) - 1, len(texts) - 1)}...")
    
    batch_embeddings = []  # Здесь будут эмбеддинги текущей порции
    text_counter = 0  # Счётчик текстов внутри батча
    for text in batch:
        text_counter += 1
        if text_counter % 10 == 0:
            print(f"  Обработано {text_counter} текстов в текущем батче")
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                timeout=30
            )
            batch_embeddings.append(response.data[0].embedding)
        except Exception as e:
            print(f"Ошибка для текста: {e}")
            # Заглушка при ошибке (вектор из нулей)
            batch_embeddings.append([0.0] * 1536)
    
    # Добавляем обработанную порцию к общему массиву
    embeddings_array.extend(batch_embeddings)
    
    # Сохраняем прогресс в кеш
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(embeddings_array, f)
    print(f"Сохранено {len(embeddings_array)} эмбеддингов")


print(f"Готово: сгенерировано {len(embeddings_array)} эмбеддингов")


# Проверка 1: совпадение длин текстов и эмбеддингов
if len(texts) != len(embeddings_array):
    print(f"Ошибка: число текстов ({len(texts)}) не совпадает с числом эмбеддингов ({len(embeddings_array)}).")
    exit(1)

# Проверка 2: размер эмбеддинга (для text-embedding-3-small это 1536)
if embeddings_array:
    emb_size = len(embeddings_array[0])
    if emb_size != 1536:
        print(f"Ошибка: размер эмбеддинга {emb_size}, ожидается 1536.")
        exit(1)

# Проверка 3: нет ли пустых текстов или эмбеддингов
empty_texts = [i for i, t in enumerate(texts) if not t.strip()]
if empty_texts:
    print(f"Предупреждение: найдены пустые тексты на позициях {empty_texts}. Удаляем...")
    # Фильтруем пустые
    texts = [t for t in texts if t.strip()]
    embeddings_array = [emb for i, emb in enumerate(embeddings_array) if i not in empty_texts]


if not texts or not embeddings_array:
    print("Ошибка: после фильтрации не осталось валидных данных.")
    exit(1)


# Путь к сохранённому индексу FAISS
FAISS_INDEX_PATH = "./faiss_index"

# Проверка: существует ли сохранённый индекс?
if os.path.exists(FAISS_INDEX_PATH):
    # Загружаем существующий индекс
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings=None  # если эмбеддинги уже в индексе
    )
    print("Загружено существующее FAISS-хранилище.")
else:
    # Создаём новое хранилище ТОЛЬКО если есть документы для добавления
    if documents:  # если список документов не пуст
        vectorstore = FAISS.from_embeddings(
                                            texts=texts,
                                            embeddings=embeddings_array,
                                            embedding=None
                                            )
        print("Создано новое FAISS-хранилище с документами.")
    else:
        # Если документов нет — создаём "пустышку" через обходной путь
        # (FAISS требует хотя бы 1 вектор для инициализации)
        dummy_embedding = [0.0] * 1536  # фиктивный эмбеддинг
        dummy_doc = Document(page_content="", metadata={"id": "dummy"})
        vectorstore = FAISS.from_embeddings(
                                            texts=texts,
                                            embeddings=embeddings_array,
                                            embedding=None
                                            )
        # Удаляем фиктивный документ сразу после создания
        vectorstore.delete(ids=["dummy"])
        print("Создано пустое FAISS-хранилище (через обходной путь).")

# Удаляем старые версии документов с теми же ID
doc_ids = [doc.metadata["id"] for doc in documents]
existing_docs = vectorstore.get(ids=doc_ids)

existing_ids = []
if "documents" in existing_docs and existing_docs["documents"]:
    existing_ids = [doc.metadata["id"] for doc in existing_docs["documents"]]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
        print(f"Удалено {len(existing_ids)} документов с повторяющимися ID.")

else:
    print("Нет существующих документов для удаления.")

# Добавляем новые документы с эмбеддингами
print(f"Добавляем {len(documents)} новых документов...")
vectorstore.add_embeddings(
    embeddings=embeddings_array,  # список списков чисел (без numpy!)
    documents=documents            # список Document
)

# Дополнительная проверка: нет ли уже документов с такими ID?
# (FAISS не гарантирует уникальность ID, поэтому проверяем вручную)
existing_after = vectorstore.get(ids=doc_ids)
if existing_after and existing_after["documents"]:
    print("Предупреждение: документы с такими ID уже присутствуют в хранилище.")
else:
    print("Все документы успешно добавлены.")

# Сохраняем обновлённое хранилище на диск
print("Сохраняем FAISS-хранилище на диск...")
vectorstore.save_local(FAISS_INDEX_PATH)
print("Готово! FAISS-хранилище обновлено и сохранено.")