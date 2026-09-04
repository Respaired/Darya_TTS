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

<center><h1>The Poor man's TTS</h1></center>

| | |
|---|---|
| Architecture | Rectified-flow encoder/decoder transformer, 1B params |
| Objective | Spanned mask (infilling) |
| Tokenizer | FSQ latents @ 12.5 Hz |
| Max sequence | 30s (prompt + output combined) |
| Languages | English, Persian (+Tajik), Russian |

## Inference

Start with the [inference notebook](https://huggingface.co/Respair/Darya_TTS/blob/main/inference_notebook.ipynb).

## What it does

- **Fast on CPU**, even at 1B, thanks in part to a few inference-side tricks and its efficient speech tokenizer.
- **Style conditioning** alongside the usual audio prompting path. This frees the whole context window for your input text, and makes prompting language-agnostic.
- **Prompt mixing** by mixing speaker vectors you can create new speaker attributes.
- **Speech editing.**
- **Multispeaker generation** via control tags (`<S1>`, `<S2>` etc.), disfluencies (`uh`, `umm`), and non-speech sounds through supported emojis.
- **Phoneme-level Persian, Tajik and Russian support.**
- **Cheap and easy to train.**

## Speed

At 16 steps, Darya reaches an RTF of ~0.05 on a high-end server CPU (Arm Neoverse V2 or similar), or ~0.5 on an i7-12700H laptop. Dropping to 8 steps gets you ~0.25 on the 12700H, though that's pushing it, the model isn't distilled.

Tested on RTX 5090, 3090 and H100. Your mileage will vary with hardware, but everything included here earns its place.

The goal of the project was the fastest modern speech synthesizer possible on a limited budget, without shrinking the model to get there. Investing in a good 12.5 Hz tokenizer is part of what made that work. (1B was chosen because I had the headroom; on cheaper GPUs, 500M makes more sense.)

## Languages

The focus this time was **Persian** and **Tajik**, with some **Russian** (best effort). **English** is also supported.

## Training

You need a dataset with pre-extracted Dune FSQ latents and text labels, plus any tokenizer `AutoTokenizer` can load , so two columns, `latents` and `text`. For the second stage, add an `audio` column, since TitaNet needs something to extract speaker latents from.

```bash
accelerate launch --mixed_precision bf16 train.py \
    --config config_transformer.json \
    --dataset /your_dataset \
    --exp_dir exp/darya \
    --tokenizer "your/tokenizer"
```

The second stage and its adversarial training are both optional. I never enabled the discriminator myself , too expensive to be worth it.

**Training details:** a single H100, effective batch size of 288, cosine schedule, roughly 400k steps over about a week of actual training. It wasn't one smooth run; I was trying things out, retraining and tweaking checkpoints to learn and unlearn various behaviours along the way. Some of that history is probably baked into what's uploaded here.

## Limitations / Notes

**Sequence length.** A regular audio prompt uses the infilling path, so it lives in the same sequence as your output. The model was trained on 30s chunks, which means prompt length plus output length has to fit inside that budget. TitaNet speaker latents don't have this constraint, but speaker similarity won't be as strong.

**Prosody depends on the length prior.** Pronunciation errors do too. There's no clean answer here: either you train the model to predict pads and silence and pay the overhead, or you lean on heuristics and length predictors that are sometimes sub-optimal. Your best available control is punctuation , use it liberally.

**Audio prompts are optional but recommended.** Darya has a full text encoder and learns alignment implicitly, so it will generate without a prompt. But it was trained on the spanned mask objective, which is the best objective we currently have for voice similarity, so it does better with one.

**Persian compounds.** Morakkab words (نرم افزار, آب جوش) and missing ezafe can come out wrong. in that case please try fixing the Finglish, add or remove spaces  and regenerate with a different seed. You are guaranteed to get what you want, just maybe not on the first try.

**Multispeaker outside English isn't robust yet.** That's a data distribution problem, and I will fix it at some point.

note: 
Persian is really the bane of my existence. the script and the absence of a good ASR model both work against you. I developed a transliteration pipeline with real human annotated data over the past few months, so Darya works with Finglish, and I've provided a model that converts Persian text to its transliteration. <br> It isn't bulletproof, but it gives you full control over generation, and with correct Finglish input, pronunciation should be near flawless.

---

I hope it's useful. Let me know if you have questions.