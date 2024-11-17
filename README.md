# SandalWood Guide
Our project leverages state-of-the-art language models and speech recognition technologies to create an intelligent assistant that can understand and respond to user queries in both Kannada and English. Using a dataset of Kannada audio files, we train IndicWhisper to convert the audio to Kannada text, which is then translated into English using the NLLB model. This allows us to store valuable content in a multilingual dataset. The system is designed to interact with users, processing their spoken questions in Kannada, converting the audio to text, translating it to English, and then retrieving the relevant answers from the stored dataset using Grok. The final responses are provided back to the user in both Kannada and English.

# Features
1) Speech-to-Text Conversion: Utilizes IndicWhisper to convert Kannada audio input into text.
2) Language Translation: Employs the NLLB model to translate Kannada text into English for cross-lingual understanding.
3) Dataset Creation: The Kannada-to-English converted text is stored in a dataset for future querying.
4) User Interaction: Users can ask questions in Kannada, which are converted to Kannada text and then translated into English.
5) Answer Retrieval: Groq is used to find the most relevant answers from the dataset based on the translated query.
6) Multilingual Output: Answers are provided back to users in both Kannada and English, ensuring accessibility in both languages.

# Technologies Used
1)Python

2)Flask: For web-based interaction and user interface.

3)Groq for finding answers from dataset.

4)IndicWhisper for converting Speech to Text.

5)NLLB for translating English to Kannada and vice versa.

# Installation:

1)Download the model from :(https://iiitbac-my.sharepoint.com/:u:/g/personal/pranay_kelotra_iiitb_ac_in/EX87I__EZbRKqSqokJLqL7cBH89rssVH6mepiZu9TzSdbg?e=kVqBMA)
# Prerequisites
  - Python 3.11

# Project Setup:

1)Clone the repository using :

-  git clone https://github.com/YashGupta2106/ML_Fiesta

2)cd project-name

3)Install the dependencies:

-  pip install -r requirements.txt

4)Set up the Groq API by configuring your API key.


# Required libraries:

  -pip install Flask werkzeug torch transformers librosa soundfile requests groq

# Usage:

1)Run the Flask Application:

-  python app.py

2)Navigate to the local host link generated in the terminal.

# Contributing:
Feel free to submit issues or pull requests to improve the project.
  
# Contributors
Pranay Kelotra, Ansh Gupta, Yash Gupta
