---
license: openrail++
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

<center><h1>The Poor man's TTS</h1></center>

| | |
|---|---|
| Architecture | Rectified-flow encoder/decoder transformer |
| Objective | Spanned mask (infilling) |
| Audio shape | FSQ latents @ 12.5 Hz - 44.1khz |
| Size | 1B params |
| Languages | English, Persian (+Tajik), Russian |

## What is this?
Darya is a fat, but fast speech generation neural net that can be trained cheaply, easily and you don't have to compromise much on its capacity.

## Inference

Start with the [inference notebook](https://huggingface.co/Respair/Darya_TTS/blob/main/inference_notebook.ipynb).

or the [gradio space](https://huggingface.co/spaces/Respair/Darya) - the denoiser is quantized to 8 bit, which causes degradation.

## Features

- **Fast on CPU**, even at 1B, thanks in part to a few inference-side tricks and its efficient speech tokenizer.
- **Style conditioning** alongside the usual audio prompting path. This frees the whole context window for your input text, and makes prompting language-agnostic.
- **Prompt mixing** by mixing speaker vectors you can create new speaker attributes.
- **Speech editing.**
- **Multispeaker generation** via control tags (`<S1>`, `<S2>` etc.), disfluencies (`uh`, `umm`), and non-speech sounds through supported emojis.
- **Phoneme-level Persian, Tajik and Russian support.**
- **Possibly the largest Zero-shot Persian model out there**
- **Cheap and easy to train.**

## Details

The goal of this project was to see if I could develop the fastest modern speech synthesizer possible (especially on cpu) on a limited budget, without compromising on the model size. <br> 

for more details please check [here](https://huggingface.co/Respair/Darya_TTS/blob/main/FootNotes_Limitations.md).

## Training

You need a dataset with pre-extracted Dune FSQ latents and text, plus any tokenizer `AutoTokenizer` can load.
first [extract the latents](https://huggingface.co/Respair/dune_codec/blob/main/dune_extraction.py) then train.

```bash
accelerate launch --mixed_precision bf16 train.py \
    --config config_transformer.json \
    --dataset /your_dataset \
    --exp_dir exp/darya \
    --tokenizer "your/tokenizer"
```

The second stage and its adversarial component are both optional. I never enabled the discriminator myself, too expensive to be worth it.

If you want to use another codec, you can just change dim 52 to your target.
but beware that this may cost you a big chunk of the efficiency gains that this model offers.


## License
see [LICENSE.md](https://huggingface.co/Respair/Darya_TTS/blob/main/LICENSE.md).

---
I hope this work proves to be useful to you. Let me know if you have questions (preferably on X / twitter or email)

Specal thanks to my good friend [Muhtasham](https://huggingface.co/muhtasham) for his financial support and his work on Tajik. <br>
and also [Mahdi](https://huggingface.co/Mahdimef) and [Amir](https://huggingface.co/eapakJR) for their help; [Den4ik](https://github.com/Den4ikAI) for `ruaccent`.