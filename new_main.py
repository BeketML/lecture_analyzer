import os
import whisperx
import pandas as pd
import warnings
import torch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")
load_dotenv()

os.environ["PATH"] += r";C:\Users\ASUS\ffmpeg\bin"

# Пути к файлам
audio_file_path = r"C:\Users\ASUS\desktop\new_stt\test.mp3"
syllabus_file_path = r"C:\Users\ASUS\desktop\new_stt\syllabus.txt"

hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not os.path.exists(audio_file_path) or not os.path.exists(syllabus_file_path):
    print("Ошибка: отсутствует файл лекции или силлабуса.")
    exit()

# Загрузка силлабуса
with open(syllabus_file_path, "r", encoding="utf-8") as f:
    syllabus_text = f.read()

class AudioProcessor:
    """Обрабатывает аудио, переводит в текст и анализирует лекцию"""
    
    def __init__(self, device="cpu", batch_size=16, compute_type="int8"):
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        try:
            self.model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=self.device)
        except Exception as e:
            print(f"Ошибка загрузки моделей: {e}")
            raise

    def transcribe_audio(self, audio_file):
        """Распознает речь из аудиофайла"""
        try:
            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            return result
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")
            return None

    def convert_to_df(self, result): 
        """Конвертирует результат в DataFrame"""
        if result is None:
            print("Ошибка: результат транскрипции отсутствует.")
            return None
        try:
            df = pd.DataFrame(result["segments"], columns=["start", "end", "text", "speaker"])
            return df
        except Exception as e:
            print(f"Ошибка конвертации в DataFrame: {e}")
            return None

class LectureAnalyzer:
    """Анализирует содержание лекции"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
        self.prompt_template = """
        Ты анализируешь лекцию. Твои задачи:
        1. Определи, соответствует ли лекция теме, заданной в силлабусе.
        2. Найди любые отклонения от темы (например, личные рассказы лектора).
        3. Оцени активность студентов по числу их реплик.

        **Силлабус:**  
        {syllabus}

        **Лекция:**  
        {lecture_text}

        **Ответ:**  
        - Лекция соответствует теме? (Да/Нет)  
        - Какие отклонения от темы?  
        - Насколько активны студенты (1-10)?
        """

        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["syllabus", "lecture_text"])
        self.output_chain = self.prompt | self.llm | StrOutputParser()

    def analyze_lecture(self, syllabus, lecture_text):
        """Оценивает лекцию по заданным критериям"""
        return self.output_chain.invoke({"syllabus": syllabus, "lecture_text": lecture_text})

# Запуск обработки
processor = AudioProcessor()
result = processor.transcribe_audio(audio_file_path)
df = processor.convert_to_df(result)

if df is None or df.empty:
    print("Ошибка: Датафрейм пустой, нечего анализировать.")
    exit()

# Сохраняем текст лекции
df.to_csv("lecture_transcription.csv", index=False, encoding="utf-8")
print("Транскрипция лекции сохранена в 'lecture_transcription.csv'.")

# Анализ лекции
analyzer = LectureAnalyzer()
lecture_text = " ".join(df["text"].tolist())
analysis = analyzer.analyze_lecture(syllabus_text, lecture_text)

# Сохраняем анализ
with open("lecture_analysis.txt", "w", encoding="utf-8") as f:
    f.write(analysis)

print("Анализ лекции сохранен в 'lecture_analysis.txt'.")
