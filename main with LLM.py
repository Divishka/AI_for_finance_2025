import os
import torch
from transformers import AutoTokenizer
from openai import OpenAI, APIConnectionError, RateLimitError
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# 1. Проверка токенизатора
def test_tokenizer():
    print("\n1. Проверка токенизатора...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        sample_text = "Это тестовый текст для проверки токенизации."
        tokens = tokenizer(sample_text, return_tensors="pt")
        print(f"✓ Токенизатор загружен. Количество токенов: {tokens['input_ids'].shape[1]}")
        return tokenizer
    except Exception as e:
        print(f"✗ Ошибка токенизатора: {e}")
        return None

# 2. Проверка подключения к OpenAI API
def test_openai_api(api_key, base_url):
    print("\n2. Проверка подключения к OpenAI API...")
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="Тестовый запрос для проверки API"
        )
        embedding = response.data[0].embedding
        print(f"✓ API доступно. Получен эмбеддинг длиной {len(embedding)}")
        return client
    except APIConnectionError:
        print("✗ Ошибка подключения к API. Проверьте URL и интернет-соединение.")
    except RateLimitError:
        print("✗ Превышен лимит запросов к API.")
    except Exception as e:
        print(f"✗ Ошибка API: {e}")
    return None

# 3. Проверка генерации эмбеддингов
def test_embedding_generation(client, tokenizer):
    print("\n3. Проверка генерации эмбеддингов...")
    sample_text = "Пример текста для генерации эмбеддинга."
    try:
        # Проверка длины текста (чтобы не превысить лимит API)
        token_count = len(tokenizer(sample_text)['input_ids'])
        if token_count > 8000:
            sample_text = sample_text[:8000]
        
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=sample_text
        )
        embedding = response.data[0].embedding
        print(f"✓ Эмбеддинг сгенерирован. Длина: {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"✗ Ошибка генерации эмбеддинга: {e}")
        return None

# 4. Проверка векторного хранилища (Chroma)
def test_chromadb():
    print("\n4. Проверка векторного хранилища Chroma...")
    # Создаём тестовый документ
    test_doc = Document(
        page_content="Тестовый документ для проверки Chroma.",
        metadata={"id": "test_id", "source": "test"}
    )
    test_embedding = [0.1] * 1536  # Заглушка (в реальном случае — ваш эмбеддинг)
    
    try:
        # Создаём временное хранилище
        vectorstore = Chroma.from_embeddings(
            embeddings=[test_embedding],
            documents=[test_doc],
            embedding=None,
            persist_directory="./test_chromadb"
        )
        
        # Проверяем поиск
        results = vectorstore.similarity_search("Тестовый документ", k=1)
        if results:
            print(f"✓ Chroma работает. Найден документ: {results[0].page_content}")
        else:
            print("✗ Chroma не вернул результаты поиска.")
        
        # Очищаем тестовые файлы
        vectorstore.delete_collection()
        os.rmdir("./test_chromadb")
    except Exception as e:
        print(f"✗ Ошибка Chroma: {e}")


# Основной тест
if __name__ == "__main__":
    print("Запуск диагностики LLM‑конвейера...\n")
    
    # Шаг 1: Токенизатор
    tokenizer = test_tokenizer()
    if not tokenizer:
        exit(1)
    
    # Шаг 2: API OpenAI
    API_KEY = "sk-k4GzLvBEsBYNbtVPpDaEMg"  # Ваш API‑ключ
    BASE_URL = "https://ai-for-finance-hack.up.railway.app/"  # Ваш URL
    client = test_openai_api(API_KEY, BASE_URL)
    if not client:
        exit(1)
    
    # Шаг 3: Генерация эмбеддингов
    embedding = test_embedding_generation(client, tokenizer)
    if not embedding:
        exit(1)
    
    # Шаг 4: Chroma
    test_chromadb()
    
    print("\n✅ Все компоненты работают корректно!")