import gradio as gr
from yt_dlp import YoutubeDL
import os
import jax
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

pipeline = FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16)
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("/home/demo/source/jax_cache")
    
def download_video(url):
  ydl_opts = {'overwrites':True, 'format':'bestaudio[ext=m4a]', 'outtmpl':'/home/demo/source/audio.m4a'}
  with YoutubeDL(ydl_opts) as ydl:
    ydl.download(url)
    return f"/home/demo/source/audio.m4a"

def transcribe(audio_in):
    outputs = pipeline("/home/demo/source/audio.m4a")
    text = outputs["text"]
    return text

app = gr.Blocks()
with app:
  gr.Markdown('## openai/whisper-large-v2')
  with gr.Row():
    with gr.Column():
      input_text = gr.Textbox(show_label=False, value="https://www.youtube.com/watch?v=SN2sak8Tp70")
      input_download_button = gr.Button(value="Download from YouTube")
      input_transcribe_button = gr.Button(value="Transcribe")
    with gr.Column():
        audio_out = gr.Audio(label="Output Audio")
        text_out = gr.Textbox(label="Output Text")
    input_download_button.click(download_video, inputs=[input_text], outputs=[audio_out])
    input_transcribe_button.click(transcribe, inputs=[audio_out], outputs=[text_out])
  
app.launch()
