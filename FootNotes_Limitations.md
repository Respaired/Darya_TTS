## Details

The goal of this project was to see if I could develop the fastest modern speech synthesizer possible (especially on cpu) on a limited budget, without compromising on the model size. <br> 


- Speed:

At 16 steps, Darya reaches an RTF of ~0.05-0.09 on a high-end server CPU (Arm Neoverse V2 or similar), or ~0.5 on an i7-12700H laptop. Dropping to 8 steps gets you ~0.25 on the 12700H, though that's pushing it, the model isn't distilled.

I have already tested everything on RTX 5090, 3090, 3070, V100 and H100. your mileage will vary with hardware, but I think everything included here to increase efficieny is proven to work.
make sure to compile your model with max-autotune-no-cudagraphs.


The focus this time was **Persian** and **Tajik**, with some **Russian** (best effort). **English** is also supported.

| | |
|---|---|
| English | 22,000+ hours |
| Persian (+Tajik) | 14,000 hours |
| Russian | 3,500 hours |
| Other languages | 12,000 hours — used for robustness; not directly usable |

## Notes and some Limitations

**Sequence length.** A regular audio prompt uses the infilling path, so it lives in the same sequence as your output. The model was trained on 30s chunks, which means prompt length plus output length has to fit inside that budget. TitaNet speaker latents don't have this constraint, but speaker similarity won't be as strong.

**Prosody depends on the duration prior.** basically a good duration (sequence length) predictor impacts everything. there's no clean answer here, either you train the model to predict pads or silence and pay an extreme overhead, or you lean on heuristics and length predictors that are often meh.

**Audio prompts are optional but recommended.** Darya has a full text encoder and learns alignment implicitly, so it will generate without a prompt. But it was trained on the spanned mask objective, which is the best thing we currently have for prompt similarity, so it works better with one.

**Persian compounds.** Morakkab words (نرم افزار, آب جوش) and missing ezafe can come out wrong with the Finglishizer. in that case please try fixing the Finglish, add or remove spaces  and regenerate with a different seed. You are guaranteed to get what you want, just maybe not on the first try.

**Multispeaker outside English isn't robust yet.** That's a data distribution problem, and I will fix it at some point.

**Parameter size** 1B is what I went with, but if you decided to train from scratch, something around 500m makes sense. it's more aligned with consumer grade gpus. 



