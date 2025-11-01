import pandas as pd
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import torch
import os



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
"""
# 3. Подготовка токенизатора для подсчёта токенов
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Можно заменить на другую модель
"""
# 2. Токенизатор и разбивка на чанки
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit(1)

print("Модель загружена:", model is not None)
print("Токенизатор загружен:", tokenizer is not None)

# Функция для подсчёта токенов
def count_tokens(text):
    return len(tokenizer.encode(text))

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

"""
# 5. Сохраняем отредактированный DataFrame в новый CSV-файл
output_file_path = './data/train_data_with_chunks.csv'  # Путь к новому файлу
train_data.to_csv(output_file_path, index=False)

print(f"Отредактированные данные сохранены в файл: {output_file_path}")

# Пример вывода первых 5 строк с разбитыми текстами
print(train_data[['id', 'chunks']].head())

# Вывод полных чанков для первых 10 статей
print("\n" + "="*80)
print("ПОЛНЫЕ ЧАНКИ ДЛЯ ПЕРВЫХ 10 СТАТЕЙ")
print("="*80)

for idx, row in train_data.head(3).iterrows():
    print(f"\nID: {row['id']}")
    print(f"Количество чанков: {len(row['chunks'])}")
    print("-"*50)
    
    for i, chunk in enumerate(row['chunks'], 1):
        print(f"  Чанк {i} (длина: {count_tokens(chunk)} токенов):")
        print(f"    {chunk[:500]}...")  # Первые 500 символов + многоточие
        print()  # Пустая строка между чанками
    
    print("-"*50)
"""

# 4. Подготовка документов для LangChain
documents = []
for _, row in train_data.iterrows():
    for chunk in row['chunks']:
        doc = Document(
            page_content=chunk,
            metadata={"id": row['id'], "source": "train_data"}
        )
        documents.append(doc)

# 2. Формируем список текстов для эмбеддинга
texts = [doc.page_content for doc in documents]
# 3. Проверяем данные
print(f"Длина texts: {len(texts)}")
if texts:
    print(f"Первый текст (первые 200 символов): {texts[0][:200]}...")
else:
    print("Список texts пуст!")

def get_embeddings_batch(texts, model, tokenizer, batch_size=32):
    """
    Генерирует эмбеддинги для текстов пакетно, чтобы не перегружать память
    """
    embeddings = []
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size else 0)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Обработка пакета {i//batch_size + 1}/{total_batches} (размер: {len(batch)})...")        
        # Токенизация
        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )        
        # Инференс модели
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Используем mean pooling
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Конвертируем в список и добавляем к общему результату
        embeddings.extend(batch_embeddings.cpu().tolist())
    
    return embeddings


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # Последние скрытые состояния
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Собираем тексты для эмбеддингов
texts = [doc.page_content for doc in documents]

# Получаем эмбеддинги с пакетной обработкой
print(f"Начинаем генерацию эмбеддингов для {len(texts)} текстов...")
embeddings_array = get_embeddings_batch(
    texts=texts, 
    model=model,
    tokenizer=tokenizer,
    batch_size=32
)
print(f"Эмбеддинги сгенерированы!")

# Загружаем существующее векторное хранилище (если оно есть)
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