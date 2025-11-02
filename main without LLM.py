import pandas as pd
import numpy as np
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

if os.path.exists("chunks_cache.pkl"):
    print("Загружаем сохранённые чанки...")
    with open("chunks_cache.pkl", "rb") as f:
        cached_chunks = pickle.load(f)
    # проверяем, что файл действительно содержит колонки id и chunks
    if isinstance(cached_chunks, pd.DataFrame):
        train_data = train_data.merge(cached_chunks, on='id', how='left')
    else:
        print("⚠️ Файл chunks_cache.pkl не в формате DataFrame, пересоздаём чанки...")
        cached_chunks = None
else:
    print("Файл с чанками не найден — создаём заново...")
    chunks_list = []
    for text in train_data['cleaned_text']:
        chunks = text_splitter.split_text(text)
        chunks_list.append(chunks)
    # Добавляем разбитые тексты обратно в DataFrame
    train_data['chunks'] = chunks_list
    # Кеш чанков
    with open("chunks_cache.pkl", "wb") as f:
        pickle.dump(train_data[['id', 'chunks']], f)



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

doc_db = {}
for i, doc in enumerate(documents):
    doc_db[doc.metadata["id"]] = {
        "document": doc,
        "embedding": np.array(embeddings_array[i])  # numpy
    }

with open("doc_db.pkl", "wb") as f:
    pickle.dump(doc_db, f)

"""
pp = pprint.PrettyPrinter(indent=2)

pp.pprint(doc_db)
print(f"Всего документов в doc_db: {len(doc_db)}")
# Проверяем, все ли эмбеддинги одинаковой длины
emb_lengths = [len(data["embedding"]) for data in doc_db.values()]
print(f"Длина эмбеддингов: {set(emb_lengths)} (должно быть {emb_lengths[0]})")
# Список всех ID
print(f"Список ID: {list(doc_db.keys())}")
"""
"""
for doc_id, data in doc_db.items():
    print(f"ID: {doc_id}")
    print(f"  Metadata: {data['document'].metadata}")
    print(f"  Content length: {len(data['document'].page_content)} символов")
    print(f"  Embedding shape: {data['embedding'].shape}")
    print("-! * 50")
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# === 1. Загрузка базы документов (из предыдущего этапа) ===
with open("doc_db.pkl", "rb") as f:
    doc_db = pickle.load(f)

print(f"✅ Загружено документов: {len(doc_db)}")

# ================== 2. Вопрос пользователя СЮДА ПИСАТЬ ВОПРОС ===================
user_question = "Куда обращаться, если заемные деньги ушли мошенникам?"

# === 3. Генерация эмбеддинга для вопроса ===
emb_client = OpenAI(
    base_url="https://ai-for-finance-hack.up.railway.app/",
    api_key="sk-k4GzLvBEsBYNbtVPpDaEMg"
)

print("📊 Генерация эмбеддинга вопроса через модель для эмбендингов...")
question_emb = emb_client.embeddings.create(
    model="text-embedding-3-small",
    input=user_question
).data[0].embedding

# === 4. Поиск ближайших документов ===
# Косинусное сходство
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Считаем схожесть между вопросом и каждым документом
similarities = []
for doc_id, doc_data in doc_db.items():
    sim = cosine_similarity(question_emb, doc_data["embedding"])
    similarities.append((doc_id, sim))

# Сортируем документы по схожести и выбираем топ-5
top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

# === 5. Формируем контекст из ближайших документов ===
context_parts = []
for doc_id, sim in top_docs:
    doc = doc_db[doc_id]["document"]
    meta = doc.metadata
    block = (
        f"Текст: {doc.page_content}\n"
        f"Аннотация: {meta.get('annotation', '')}\n"
        f"Теги: {meta.get('tags', '')}"
    )
    context_parts.append(block)

context = "\n\n".join(context_parts)


# ====================== 6. Формируем промпт. СЮДА ВСТАВЛЯТЬ ПРОМТ ===========================
system_prompt = (
    "Ты финансовый ассистент. Отвечай чётко, по-русски, используя только факты из контекста. Не используй Markdown или выделения. Читай весь контект и дай единый полный ответ по присланным данным. Если в контексте нет данных для какого-либо вопроса — напиши об этом прямо, без выдумок. Но дай все равно обобщенный ответ. Не используй Markdown, звёздочки, эмодзи или форматирование. Ответ должен быть текстом, структурированным по пунктам."
)
user_prompt = f"Контекст:\n{context}\n\nВопрос:\n{user_question}\n\nОтвет:" # сюда автоматически подтягивается контект и вопрос 


# === 7. Отправляем запрос к LLM (через langchain-openai) ===
llm = ChatOpenAI(
    api_key="sk-BuwLErZ4eL4yTAjfQxLaIA",  # ключ для LLM
    base_url="https://ai-for-finance-hack.up.railway.app/",
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct",
    temperature=0.2,
    max_tokens=500
)
# === 7.1. Печатаем полный текст, который отправляем в модель ===
print("\n================= 📤 ПОЛНЫЙ ПРОМПТ, ОТПРАВЛЯЕМЫЙ В LLM =================")
print(f"\n[System message]\n{system_prompt}\n")
print(f"[User message]\n{user_prompt}\n")
print("=====================================================================\n")

print("🤖 Отправка запроса к модели...")
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt)
]

response = llm.invoke(messages)

# === 8. Выводим ответ ===
print("\n💬 Ответ модели:")
print(response.content)