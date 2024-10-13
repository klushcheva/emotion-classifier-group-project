from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel

# Инициализация FastAPI
app = FastAPI()

# ВНИМАНИЕ!!! Скопируй в директорю, в пути к которой нет кириллицы
# Путь к модели и токенизатору
model_path = r'C:\Users\Project\model'  # ВНИМАНИЕ!!! Укажи путь к директории 

# Загрузка токенизатора
tokenizer = BertTokenizer.from_pretrained(model_path)

# Определение кастомной модели
class BertSentimentAnalyzer(nn.Module):
    def __init__(self):
        super(BertSentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
        self.num_labels = 3
        self.fc_classification = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        logits = self.fc_classification(pooled_output)
        return logits

# Инициализация модели
model = BertSentimentAnalyzer()
model.load_state_dict(torch.load(f'{model_path}/pytorch_model.bin', map_location=torch.device('cpu')))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Словарь для перевода предсказания в метки тональности
label_map = {0: "негативная", 1: "нейтральная", 2: "позитивная"}

# Модель данных для входного текста
class TextInput(BaseModel):
    text: str

# Функция для предсказания тональности
def predict_sentiment(text):
    # Токенизация текста
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Удаляем 'token_type_ids', если модель его не использует
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    # Перемещаем входные данные на устройство (GPU или CPU)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Прогон через модель
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Получение предсказания
    predictions = torch.argmax(outputs, dim=-1).item()

    # Возвращаем предсказанную тональность
    return label_map[predictions]

# Маршрут для анализа текста
@app.post("/analyze")
def analyze_text(input: TextInput):
    result = predict_sentiment(input.text)
    return {"text": input.text, "sentiment": result}
