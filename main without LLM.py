import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import os
from openai import OpenAI, APIConnectionError, RateLimitError
import pickle


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
        doc = Document(
            page_content=chunk,
            metadata={"id": row['id'], "source": "train_data"}
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

EMBEDDINGS_CACHE = "./embeddings_cache.pkl"

# Проверяем, есть ли кешированные эмбеддинги
if os.path.exists(EMBEDDINGS_CACHE):
    print("Загружаем кешированные эмбеддинги...")
    with open(EMBEDDINGS_CACHE, "rb") as f:
        embeddings_array = pickle.load(f)
else:
    # Генерируем эмбеддинги через OpenAI
    print(f"Начинаем генерацию эмбеддингов для {len(texts)} текстов...")
    embeddings_array = get_openai_embeddings(texts, model="text-embedding-3-small")
    # Сохраняем в кеш
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(embeddings_array, f)
    print("Эмбеддинги сохранены в кеш")

print(f"Размер кеша: {len(embeddings_array)} эмбеддингов")

# 9. Работа с векторным хранилищем
if os.path.exists("./chromadb"):
    vectorstore = Chroma(
        persist_directory="./chromadb",
        embedding_function=None  # embedding не нужен — мы работаем с готовыми эмбеддингами
    )
else:
    # Если базы ещё нет — создаём новую
    vectorstore = Chroma.from_embeddings(
        embeddings=[],
        documents=[],
        embedding=None,
        persist_directory="./chromadb"
    )

# Удаляем старые версии документов с теми же ID
doc_ids = [doc.metadata["id"] for doc in documents]
existing_docs = vectorstore.get(ids=doc_ids)
existing_ids = [doc.metadata["id"] for doc in existing_docs["documents"]]


if existing_ids:
    vectorstore.delete(ids=existing_ids)
    print(f"Удалено {len(existing_ids)} устаревших документов")

# Добавляем обновлённые документы
print(f"Добавляем {len(documents)} новых документов...")
vectorstore.add_embeddings(
    embeddings=embeddings_array,
    documents=documents
)

# Сохраняем на диск
print("Сохраняем векторное хранилище...")
vectorstore.persist()
print("Готово! Векторное хранилище обновлено.")