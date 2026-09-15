## Notes and some Limitations

**Sequence length.** A regular audio prompt uses the infilling path, so it lives in the same sequence as your output. The model was trained on 30s chunks, which means prompt length plus output length has to fit inside that budget. TitaNet speaker latents don't have this constraint, but speaker similarity won't be as strong.

**Prosody depends on the duration prior.** basically a good duration (sequence length) predictor impacts everything. there's no clean answer here, either you train the model to predict pads or silence and pay an extreme overhead, or you lean on heuristics and length predictors that are often meh.

**Audio prompts are optional but recommended.** Darya has a full text encoder and learns alignment implicitly, so it will generate without a prompt. But it was trained on the spanned mask objective, which is the best thing we currently have for prompt similarity, so it works better with one.

**Persian compounds.** Morakkab words (نرم افزار, آب جوش) and missing ezafe can come out wrong with the Finglishizer. in that case please try fixing the Finglish, add or remove spaces  and regenerate with a different seed. You are guaranteed to get what you want, just maybe not on the first try.

**Multispeaker outside English isn't robust yet.** That's a data distribution problem, and I will fix it at some point.

**Parameter size** 1B is what I went with, but if you decided to train from scratch, something around 500m makes sense. it's more aligned with consumer grade gpus. 



