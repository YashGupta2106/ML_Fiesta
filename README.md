# ML_Fiesta
Our project leverages state-of-the-art language models and speech recognition technologies to create an intelligent assistant that can understand and respond to user queries in both Kannada and English. Using a dataset of Kannada audio files, we employ IndicWhisper to convert the audio to Kannada text, which is then translated into English using the NLLB model. This allows us to store valuable content in a multilingual dataset. The system is designed to interact with users, processing their spoken questions in Kannada, converting the audio to text, translating it to English, and then retrieving the relevant answers from the stored dataset using Grok. The final responses are provided back to the user in both Kannada and English.

# Features
1) Speech-to-Text Conversion: Utilizes IndicWhisper to convert Kannada audio input into text.
2) Language Translation: Employs the NLLB model to translate Kannada text into English for cross-lingual understanding.
3) Dataset Creation: The Kannada-to-English converted text is stored in a dataset for future querying.
4) User Interaction: Users can ask questions in Kannada, which are converted to Kannada text and then translated into English.
5) Answer Retrieval: Grok is used to find the most relevant answers from the dataset based on the translated query.
6) Multilingual Output: Answers are provided back to users in both Kannada and English, ensuring accessibility in both languages.

# Technologies Used:

-Python
-Flask: For web-based interaction and user interface.
-Grok for finding answers from dataset.
-IndicWhisper for converting Speech to Text.
-NLLB for translating English to Kannada and vice versa.

# Contributors
Pranay Kelotra, Ansh Gupta, Yash Gupta
