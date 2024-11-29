import whisper
from gtts import gTTS
def transcribe_audio_to_txt(audio_file, output_txt_file):
    # 加载 Whisper 模型（选择合适的模型大小，如 'base', 'small', 'medium', 'large'）
    model = whisper.load_model("medium")

    # 加载并处理音频文件
    result = model.transcribe(audio_file,language="en")

    # 输出转录结果
    # print("Transcription:")
    # print(result["text"])
    #
    # # 将文本写入到txt文件
    # with open(output_txt_file, 'w', encoding='utf-8') as file:
    #     file.write(result["text"])
    # print(f"文本已保存到 {output_txt_file}")
    return result["text"]
def transcribe_txt_to_audio(text,output_audio_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_audio_file)

