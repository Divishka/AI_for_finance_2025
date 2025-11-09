import openai
import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI

# Инициализация клиента
client = OpenAI(
    base_url="https://ai-for-finance-hack.up.ray.app/",
    api_key="sk-BuwLErZ4eL4yTAjfQxLaIA"
)

# Функция для получения информации о модели
def get_model_info(model_name):
    try:
        # Получаем информацию о модели
        model_details = client.models.retrieve(model_name)
        
        # Извлекаем основные параметры
        max_tokens = model_details.get('max_tokens', 'N/A')
        context_window = model_details.get('context_window', 'N/A')
        token_limit = model_details.get('token_limit', 'N/A')
        
        return {
            "Модель": model_name,
            "Максимальное количество токенов": max_tokens,
            "Окно контекста": context_window,
            "Лимит токенов": token_limit
        }
    except Exception as e:
        print(f"Ошибка при получении информации о модели: {e}")
        return None

# Список моделей для проверки
models_to_check = [
    "openrouter/mistralai/mistral-small-3.2-24b-instruct",
    "openrouter/meta-llama/llama-3-70b-instruct",
    "openrouter/x-ai/grok-3-mini",
    "openrouter/google/gemma-3-27b-it"
]

# Собираем информацию о моделях
model_info_list = []
for model in models_to_check:
    info = get_model_info(model)
    if info:
        model_info_list.append(info)

# Создаем DataFrame для удобного отображения
df = pd.DataFrame(model_info_list)

# Выводим результаты
print("Информация о моделях:")
print(df)

# Дополнительно проверяем лимиты через LLM
llm = ChatOpenAI(
    api_key="sk-BuwLErZ4eL4yTAjfQxLaIA",
    base_url="https://ai-for-finance-hack.up.ray.app/",
    model="openrouter/mistralai/mistral-small-3.2-24b-instruct"
)

try:
    # Получаем информацию о лимитах
    model_params = llm.get_model_params()
    print("\nДополнительные параметры модели:")
    print(f"Максимальный размер ответа: {model_params.max_tokens}")
    print(f"Контекстное окно: {model_params.context_window}")
    print(f"Тип токенизатора: {model_params.tokenizer}")
except Exception as e:
    print(f"Ошибка при получении параметров LLM: {e}")

# Сохраняем результаты в CSV
df.to_csv("model_limits.csv", index=False, encoding='utf-8')
print("\nДанные сохранены в model_limits.csv")
