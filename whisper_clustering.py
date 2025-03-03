import subprocess
import torch
import whisper
import wave
import contextlib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

"""
# for download
!pip install -q git+https://github.com/openai/whisper.git > /dev/null
!pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null

! pip install torch whisper numpy pandas python-dotenv scikit-learn pyannote.audio langchain-google-genai langchain-core
! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

"""

audio_path = "path to your audio"
syllabus_path = "path to your syllabus"
hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")

def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class AudioProcessor:
    def __init__(self, language='any', model_size='large', audio_path=audio_path, syllabus_path=syllabus_path):
        self.language = language
        self.model_size = model_size
        self.audio_path = audio_path
        self.syllabus_path = syllabus_path
        self.device = select_device()
        self.model_name = self.get_model_name()
        self.embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device(self.device))
    
    def get_model_name(self):
      # specified parameters for english lectures
        model_name = self.model_size
        if self.language == 'English' and self.model_size != 'large':
            model_name += '.en'
        return model_name

    def convert_audio_to_wav(self, path):
      # mp3 to wav
        if not path.endswith('.wav'):
            subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
            return 'audio.wav'
        return path

    def get_audio_duration(self, path):
      # audio and speaker duaration
        with contextlib.closing(wave.open(path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return duration

    def transcribe_audio(self, path):
      # main model for speech to text
        model = whisper.load_model(self.model_name, device=self.device)
        result = model.transcribe(path)
        return result['segments']

    def extract_speaker_embeddings(self, segments):
        embeddings = []
        for segment in segments:
            audio = Audio()
            waveform, sample_rate = audio.crop('audio.wav', Segment(segment['start'], segment['end']))
            embedding = self.embedding_model({'waveform': waveform, 'sample_rate': sample_rate})
            embeddings.append(embedding.numpy())
        return np.array(embeddings)

    def cluster_speakers(self, embeddings):
        clustering = AgglomerativeClustering(n_clusters=None, metric='euclidean', linkage='ward', distance_threshold=0.5)
        labels = clustering.fit_predict(embeddings)
        return labels

    def process_audio(self):
        path = self.convert_audio_to_wav(self.audio_path)
        duration = self.get_audio_duration(path)
        print(f"Audio duration: {duration:.2f} seconds")
        
        segments = self.transcribe_audio(path)
        embeddings = self.extract_speaker_embeddings(segments)
        labels = self.cluster_speakers(embeddings)
        
        transcript = pd.DataFrame([{**segment, 'speaker': labels[i]} for i, segment in enumerate(segments)])
        transcript.to_csv('transcript.csv', index=False)
        return transcript

class LectureAnalyzer:
    def __init__(self, syllabus_path):
        self.device = select_device()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key)
        with open(syllabus_path, "r", encoding="utf-8") as f:
            self.syllabus_text = f.read()
    
    def analyze_lecture(self, transcript):
        transcript_text = "\n".join(transcript['text'])
        prompt = f"""
        Ты анализируешь лекцию. Определи, соответствует ли она силлабусу. 
        Силлабус:
        {self.syllabus_text}
        
        Лекция:
        {transcript_text}
        
        Дай анализ соответствия.
        """
        response = self.llm.invoke(prompt)
        with open("lecture_analysis.txt", "w", encoding="utf-8") as f:
            f.write(response)
        return response

class Main:
    @staticmethod
    def run():
        processor = AudioProcessor()
        transcript = processor.process_audio()
        analyzer = LectureAnalyzer(syllabus_path)
        analysis = analyzer.analyze_lecture(transcript)
        print("Анализ лекции сохранен в lecture_analysis.txt")

if __name__ == "__main__":
    Main.run()
