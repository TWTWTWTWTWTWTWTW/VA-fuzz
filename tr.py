from trans import *
def tr(text):
    # 语音转文字后利用ASR识别语音得到文本，模拟ASR的过程
    transcribe_txt_to_audio(text, "output.mp3")
    text = transcribe_audio_to_txt("output.mp3", "output.txt")
    return text