import os
import whisperx
import pandas as pd
import warnings
import torch
import traceback
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Suppress warnings
#warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_AUDIO_PATH = "lecture.mp3"
DEFAULT_SYLLABUS_PATH = "syllabus.txt"
OUTPUT_TRANSCRIPTION = "lecture_transcription.csv"
OUTPUT_ANALYSIS = "lecture_analysis.txt"

# Try to get API tokens from environment variables first, then fallback to hardcoded values
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"Проверяю путь к файлу лекции: {DEFAULT_AUDIO_PATH}")
print(f"Проверяю путь к файлу силлабуса: {DEFAULT_SYLLABUS_PATH}")


if not os.path.exists(DEFAULT_AUDIO_PATH):
    raise FileNotFoundError(f"Ошибка: файл {DEFAULT_AUDIO_PATH} не найден.")
else:
    print(f"File exist path: {DEFAULT_AUDIO_PATH}")

if not os.path.exists(DEFAULT_SYLLABUS_PATH):
    raise FileNotFoundError(f"Ошибка: файл {DEFAULT_SYLLABUS_PATH} не найден.")
    print(f"File exist path: {DEFAULT_SYLLABUS_PATH}")


class DeviceManager:
    """Manages device selection and CUDA availability"""
    
    @staticmethod
    def get_optimal_device():
        """Determine the optimal device (CUDA or CPU) based on availability and functionality"""
        device = "cpu"
        if torch.cuda.is_available():
            try:
                # Test if CUDA actually works
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                device = "cuda"
                print(f"✅ CUDA is working properly. Using device: {device}")
                print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except Exception as e:
                print(f"⚠️ CUDA is available but encountered an error: {e}")
                print("   Falling back to CPU")
        else:
            print("ℹ️ CUDA is not available. Using CPU instead.")
        
        return device
    
    @staticmethod
    def clear_cuda_cache():
        """Clear CUDA cache if available"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                print("🧹 CUDA memory cleared")
                return True
            except Exception as e:
                print(f"⚠️ Failed to clear CUDA cache: {e}")
        return False


class AudioProcessor:
    """Processes audio files for transcription and speaker diarization"""
    
    def __init__(self, device="cpu"):
        self.device = device
        
        # Set parameters based on available device
        self.batch_size = 16 if device == "cuda" else 8
        self.compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"🔧 Initializing with: device={device}, batch_size={self.batch_size}, compute_type={self.compute_type}")
        
        try:
            print("📝 Loading WhisperX model...")
            # Handle potential CUDA errors gracefully
            try:
                self.model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
            except Exception as e:
                print(f"⚠️ Failed to load model on {self.device}: {e}")
                print("   Falling back to CPU")
                self.device = "cpu"
                self.compute_type = "int8"
                self.batch_size = 8
                self.model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
            
            print("🎙️ Loading diarization pipeline...")
            try:
                self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=self.device)
            except Exception as e:
                print(f"⚠️ Failed to load diarization model on {self.device}: {e}")
                print("   Falling back to CPU for diarization")
                self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device="cpu")
                
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(f"   Details: {traceback.format_exc()}")
            raise

    def transcribe_audio(self, audio_file):
        """Transcribe audio file with speaker diarization"""
        try:
            print("🔊 Loading audio...")
            audio = whisperx.load_audio(audio_file)
            
            print("🎯 Transcribing with WhisperX...")
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            print(f"🌐 Detected language: {result['language']}")
            
            print("⚙️ Loading alignment model...")
            # Use CPU for alignment if CUDA has issues
            align_device = self.device
            try:
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=align_device)
            except Exception as e:
                print(f"⚠️ Falling back to CPU for alignment due to error: {e}")
                align_device = "cpu"
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=align_device)
            
            print("📊 Aligning segments...")
            result = whisperx.align(result["segments"], model_a, metadata, audio, align_device, return_char_alignments=False)
            
            print("👥 Running speaker diarization...")
            diarize_segments = self.diarize_model(audio)
            
            print("🔄 Assigning speakers to words...")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            print("✅ Transcription complete")
            return result
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            print(f"   Details: {traceback.format_exc()}")
            return None

    def convert_to_df(self, result): 
        """Convert transcription result to DataFrame"""
        if result is None:
            print("❌ Error: transcription result is missing.")
            return None
        try:
            df = pd.DataFrame(result["segments"])
            # Ensure all necessary columns exist
            required_columns = ["start", "end", "text", "speaker"]
            for col in required_columns:
                if col not in df.columns:
                    print(f"⚠️ Warning: '{col}' column missing from transcription result")
                    if col == "speaker":
                        df["speaker"] = "UNKNOWN"
            
            # Select only the required columns
            df = df[required_columns]
            return df
        except Exception as e:
            print(f"❌ Error converting to DataFrame: {e}")
            print(f"   Details: {traceback.format_exc()}")
            return None


class LectureAnalyzer:
    """Analyzes lecture content using AI"""

    def __init__(self, api_key=None):
        self.api_key = api_key or GEMINI_API_KEY
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.api_key)
            print("✅ Initialized Gemini model successfully")
        except Exception as e:
            print(f"❌ Error initializing Gemini model: {e}")
            raise
            
        self.prompt_template = """
        Ты анализируешь лекцию. Твои задачи:
        1. Определи, соответствует ли лекция теме, заданной в силлабусе.
        2. Найди любые отклонения от темы (например, личные рассказы лектора).
        3. Оцени активность студентов по числу их реплик.
        4. Выдели ключевые концепции и термины, упомянутые в лекции.
        5. Оцени ясность изложения материала.

        **Силлабус:**  
        {syllabus}

        **Лекция:**  
        {lecture_text}

        **Ответ:**  
        - Лекция соответствует теме? (Да/Нет)  
        - Какие отклонения от темы были замечены?  
        - Насколько активны студенты (1-10)?
        - Ключевые концепции и термины:
        - Ясность изложения (1-10):
        - Рекомендации по улучшению:
        """

        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["syllabus", "lecture_text"])
        self.output_chain = self.prompt | self.llm | StrOutputParser()

    def analyze_lecture(self, syllabus, lecture_text):
        """Evaluate lecture against syllabus and criteria"""
        try:
            return self.output_chain.invoke({"syllabus": syllabus, "lecture_text": lecture_text})
        except Exception as e:
            print(f"❌ Error analyzing lecture: {e}")
            print(f"   Details: {traceback.format_exc()}")
            return f"Error analyzing lecture: {e}"


class LectureProcessingPipeline:
    """End-to-end pipeline for lecture processing"""
    
    def __init__(self, audio_path=DEFAULT_AUDIO_PATH, syllabus_path=DEFAULT_SYLLABUS_PATH):
        self.audio_path = audio_path
        self.syllabus_path = syllabus_path
        self.device = DeviceManager.get_optimal_device()
        self.processor = None
        self.analyzer = None
        self.syllabus_text = None
        self.df = None

    def validate_files(self):
        """Validate that required files exist"""
        if not os.path.exists(self.audio_path):
            print(f"❌ Error: Audio file not found: {self.audio_path}")
            return False
        
        if not os.path.exists(self.syllabus_path):
            print(f"❌ Error: Syllabus file not found: {self.syllabus_path}")
            return False
            
        return True
        
    def load_syllabus(self):
        """Load syllabus from file"""
        try:
            with open(self.syllabus_path, "r", encoding="utf-8") as f:
                self.syllabus_text = f.read()
            print(f"✅ Syllabus loaded from {self.syllabus_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading syllabus: {e}")
            return False
            
    def initialize_components(self):
        """Initialize processor and analyzer"""
        try:
            # Clear CUDA cache before initializing
            DeviceManager.clear_cuda_cache()
            
            self.processor = AudioProcessor(device=self.device)
            self.analyzer = LectureAnalyzer()
            return True
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            print(f"   Details: {traceback.format_exc()}")
            return False
            
    def process_audio(self):
        """Process audio file and convert to DataFrame"""
        try:
            result = self.processor.transcribe_audio(self.audio_path)
            self.df = self.processor.convert_to_df(result)
            
            if self.df is None or self.df.empty:
                print("❌ Error: Empty transcription result")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            print(f"   Details: {traceback.format_exc()}")
            return False
            
    def save_transcription(self, output_path=OUTPUT_TRANSCRIPTION):
        """Save transcription to CSV file"""
        try:
            self.df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"✅ Transcription saved to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving transcription: {e}")
            return False
            
    def analyze_content(self):
        """Analyze lecture content"""
        try:
            lecture_text = " ".join(self.df["text"].tolist())
            analysis = self.analyzer.analyze_lecture(self.syllabus_text, lecture_text)
            return analysis
        except Exception as e:
            print(f"❌ Error analyzing content: {e}")
            print(f"   Details: {traceback.format_exc()}")
            return None
            
    def save_analysis(self, analysis, output_path=OUTPUT_ANALYSIS):
        """Save analysis to text file"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(analysis)
            print(f"✅ Analysis saved to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            return False
            
    def run(self):
        """Run the complete pipeline"""
        print("🚀 Starting lecture processing pipeline...")
        
        if not self.validate_files():
            return False
            
        if not self.load_syllabus():
            return False
            
        if not self.initialize_components():
            return False
            
        if not self.process_audio():
            return False
            
        if not self.save_transcription():
            return False
            
        analysis = self.analyze_content()
        if analysis is None:
            return False
            
        if not self.save_analysis(analysis):
            return False
            
        print("✅ Pipeline completed successfully")
        return True
        
    def cleanup(self):
        """Clean up resources"""
        try:
            # Delete model references
            if self.processor:
                if hasattr(self.processor, 'model'):
                    del self.processor.model
                if hasattr(self.processor, 'diarize_model'):
                    del self.processor.diarize_model
            
            # Clear CUDA memory
            DeviceManager.clear_cuda_cache()
            
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process lecture audio and analyze content")
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH, help="Path to audio file")
    parser.add_argument("--syllabus", default=DEFAULT_SYLLABUS_PATH, help="Path to syllabus file")
    parser.add_argument("--output-transcription", default=OUTPUT_TRANSCRIPTION, help="Path to save transcription")
    parser.add_argument("--output-analysis", default=OUTPUT_ANALYSIS, help="Path to save analysis")
    
    args = parser.parse_args()
    
    pipeline = LectureProcessingPipeline(args.audio, args.syllabus)
    
    try:
        success = pipeline.run()
        if success:
            print("🎉 Processing completed successfully")
        else:
            print("⚠️ Processing completed with errors")
    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        print(f"   Details: {traceback.format_exc()}")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()



# Как использовать улучшенную программу для анализа лекций
# Вот инструкция по использованию программы:
# Подготовка файлов

# Подготовьте аудиофайл лекции (например, в формате MP3)
# Создайте текстовый файл с силлабусом (учебным планом)

# Запуск программы
# Базовый запуск (с путями по умолчанию)
# bashCopypython lecture_analyzer.py
# По умолчанию программа будет искать файлы:

# lecture.mp3 - аудиофайл лекции
# syllabus.txt - файл силлабуса

# Запуск с указанием путей к файлам
# bashCopypython lecture_analyzer.py --audio "путь/к/лекции.mp3" --syllabus "путь/к/силлабусу.txt"
# Полные параметры запуска
# bashCopypython lecture_analyzer.py --audio "путь/к/лекции.mp3" --syllabus "путь/к/силлабусу.txt" --output-transcription "транскрипция.csv" --output-analysis "анализ.txt"
# Параметры командной строки

# --audio: путь к аудиофайлу лекции
# --syllabus: путь к файлу силлабуса
# --output-transcription: имя файла для сохранения транскрипции (CSV-формат)
# --output-analysis: имя файла для сохранения анализа лекции (текстовый формат)

# Результаты работы
# После успешного выполнения программа создаст два файла:

# lecture_transcription.csv - транскрипция лекции с разделением по спикерам
# lecture_analysis.txt - анализ содержания лекции

# Требования
# Для работы программы необходимы:

# Python 3.8 или выше
# Установленные библиотеки: whisperx, pandas, torch, langchain_google_genai, dotenv
# API-ключи для Hugging Face и Gemini API

# Настройка окружения
# Рекомендуется создать файл .env в той же директории с программой и добавить в него API-ключи:
# CopyHF_TOKEN=ваш_ключ_huggingface
# GEMINI_API_KEY=ваш_ключ_gemini
# Если файл .env отсутствует, программа будет использовать значения по умолчанию из кода.
# Оптимизация производительности

# Если у вас есть GPU с поддержкой CUDA, программа автоматически использует его для ускорения обработки
# При возникновении проблем с CUDA, программа автоматически переключится на CPU
# Для длинных аудиозаписей процесс может занять значительное время, особенно на CPU

# Программа выводит детальные сообщения о ходе выполнения, что поможет отследить прогресс и обнаружить возможные проблемы.