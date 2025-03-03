# Audio Speaker Identification and Lecture Analysis

## Overview
This project provides an end-to-end pipeline for speaker diarization and lecture content analysis using two different approaches:

1. **WhisperX-based Method**: Uses WhisperX for speech-to-text transcription and speaker diarization.
2. **Whisper + Clustering Method**: Uses OpenAI Whisper for transcription and agglomerative clustering for speaker identification.

The extracted transcripts are then analyzed using an LLM (Gemini) to compare lecture content with a syllabus and evaluate student engagement.

---

## Features
- **Automatic Speech-to-Text**: Converts audio lectures into text.
- **Speaker Diarization**: Identifies different speakers in the lecture.
- **Comparison with Syllabus**: Checks whether the lecture content aligns with the syllabus.
- **Student Engagement Analysis**: Evaluates student participation based on speech frequency.
- **CUDA/CPU Support**: Allows selection between CUDA (GPU) and CPU for processing.

---

## Installation
To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

Ensure you have **FFmpeg** installed and added to your system path for audio processing.

---

## Usage

### 1. Prepare Your Environment
Set up your environment variables in a `.env` file:

```
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
```

Ensure your paths are set correctly:
```python
audio_path = "path to your audio"
syllabus_path = "path to your syllabus"
```

### 2. Run the Script
Run the main processing script:
```bash
python main.py
```
The results, including the transcribed text and analysis, will be saved in a `.txt` file.

---

## Methods Used
### **1. WhisperX-Based Approach**
- Uses **WhisperX** for high-quality transcription.
- Utilizes **WhisperX's built-in speaker diarization** model for accurate speaker detection.

### **2. Whisper + Clustering Approach**
- Uses **OpenAI Whisper** for transcription.
- Extracts **speaker embeddings** using `speechbrain/spkrec-ecapa-voxceleb`.
- Applies **Agglomerative Clustering** to group similar speaker embeddings.

---

## Output
- **Transcription File**: A `.txt` file containing the full lecture transcript with speaker labels.
- **Analysis Report**: Compares lecture content with the syllabus and provides insights into student participation.

---

## Dependencies
Ensure you have the following installed:
- Python 3.8+
- WhisperX
- OpenAI Whisper
- Pyannote Audio
- LangChain
- Google Gemini API

For a full list, see `requirements.txt`.

---

## Future Improvements
- Enhance speaker diarization accuracy.
- Improve syllabus comparison using advanced NLP techniques.
- Integrate real-time processing capabilities.

---

## License
This project is licensed under the MIT License.
