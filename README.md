---
sdk: gradio
sdk_version: 3.32.0
app_file: app.py
---

## Links

🐣 Please follow me for new updates https://twitter.com/camenduru <br />
🔥 Please join our discord server https://discord.gg/k5BwmmvJJU

### Main Repo
https://github.com/sanchit-gandhi/whisper-jax

### Model
https://huggingface.co/openai/whisper-large-v2

### API Tutorial
![Screenshot 2023-06-01 100159](https://github.com/camenduru/whisper-jax-lambda/assets/54370274/7ffba440-6284-424f-908f-ec265e943cf9)

```py
from IPython.display import HTML, display
def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

!pip install gradio_client

from gradio_client import Client
client = Client("https://demo.PAGE_ID.lambdaspaces.com/")

video = client.predict("https://www.youtube.com/watch?v=SN2sak8Tp70", fn_index=0)
text = client.predict(video,	fn_index=1)
print(text)
```
