
---
license: cc-by-4.0
language:
- fa
- ru
- en
tags:
- persian
- english
- tts
- speech
- russian
---

<center>Darya: a 1B TTS that runs reasonably fast on your laptop's CPU!</center>

# Inference 

Please take a look at the [Inference notebook](https://huggingface.co/Respair/Darya_TTS/blob/main/inference_notebook.ipynb)<br>

# Details

Darya is a capable rectified flow 1B encoder / decoder transformer, trained with spanned mask objective on 50,000 hours of multilingual data. <br>

it is / has: 

  - incredibly fast even on CPU despite its size. thanks in part to a few inference-side tricks I added.
  - style conditioning alongside the usual audio prompting pass, allowsing you to save the precious context entirely for your input text, and also enabling language-agnostic prompting
  - speech editing
  - multispeaker generation + non-speech sounds
  - relatively cheap to train (because of the 12.5hz tokenizer I )



