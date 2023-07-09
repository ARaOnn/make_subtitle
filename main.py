import whisper
import stable_whisper
import os
import gc
import torch
import sys
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox

def make_subtitle(model_size, language = 'en', dir_path=None):
    global file
    model = stable_whisper.load_model(model_size, "cuda" if torch.cuda.is_available() else "cpu")
    if dir_path is None:
        dir_path = './videos/'
        
        for (root, directories, files) in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                fileName = Path(file).resolve().stem            

                audio = whisper.load_audio(file_path)
                result = model.transcribe(audio, language=language)

                outPath = f"./outputs/{fileName}.srt"
                result.to_srt_vtt(outPath, word_level=False)

                del audio
                del result

        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:        
        fileName = os.path.splitext(os.path.basename(file.name))[0]
        audio = whisper.load_audio(file.name)
        result = model.transcribe(audio, language = language)

        outPath = f"./outputs/{fileName}.srt"
        result.to_srt_vtt(outPath, word_level=False)

        del audio
        del result

        del model
        gc.collect()
        torch.cuda.empty_cache()


model_size = "large-v2"

file = filedialog.askopenfile(initialdir="videos",\
                 title = "파일을 선택 해 주세요",\
                    filetypes=(('mp4 files', '*.mp4'), ('all files', '*.*')))
if file is None:
    make_subtitle(model_size, 'en', None)
    sys.exit()

make_subtitle(model_size, 'ko', file)
