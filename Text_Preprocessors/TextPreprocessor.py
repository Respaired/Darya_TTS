import re
from dataclasses import dataclass
from typing import Literal, Optional

from llama_cpp import Llama
from ruaccent import RUAccent
from Text_Preprocessors.ruphon import RUPhon


Mode = Literal["all", "persian", "russian"]

# VERB_ENDINGS = (
#     "م", "ی", "د",
#     "یم", "ید", "ند",
#     "ام", "ای", "است", "ایم", "اید", "اند",
#     "بود", "بودم", "بودی", "بودند", "بودیم", "بودید",
#     "شد", "شدم", "شدی", "شدند", "شدیم", "شدید",
#     "کرد", "کردم", "کردی", "کردند", "کردیم", "کردید",
#     "دارد", "دارم", "داری", "دارند", "داریم", "دارید",
#     "می‌شود", "می‌شوند", "می‌کنم", "می‌کنی", "می‌کند",
#     "می‌کنیم", "می‌کنید", "می‌کنند",
# )

# AUX_VERBS = {
#     "است", "هست", "بود", "شد", "شده", "کرد", "کرده",
#     "داد", "داده", "دادند", "رفت", "رفته", "آمد", "آمده",
#     "داشت", "داشته", "دارد", "دارند", "می‌شود", "می‌شوند",
# }

# DISALLOWED_PERSIAN_SYMBOL_RE = re.compile(
#     r"[^A-Za-z0-9۰-۹٠-٩\u0600-\u06FF\u200c\s،.!؟]"
# )

# BOUNDARY_SYMBOL_RE = re.compile(r"\s*[:؛;/\\|]+\s*")

# def _is_persian_verb(self, word: str) -> bool:
#     word = word.strip(" .!؟،")
#     if not word:
#         return False

#     if word in self.AUX_VERBS:
#         return True

#     if word.startswith(("می‌", "نمی‌", "می", "نمی")):
#         return True

#     if word.endswith(("ه‌ام", "ه‌ای", "ه‌است", "ه‌ایم", "ه‌اید", "ه‌اند")):
#         return True

#     return any(word.endswith(e) for e in self.VERB_ENDINGS)

# def _fix_commas_after_verbs(self, text: str) -> str:
#     def repl(m):
#         word = m.group(1)
#         return word + "." if self._is_persian_verb(word) else word + "،"

#     return re.sub(r"([\u0600-\u06FF\u200c]+)\s*،", repl, text)

# def _normalize_persian(self, text: str) -> str:
#     if not text or not self.FA_RE.search(text):
#         return text

#     text = text.replace("\u064a", "ی").replace("\u0643", "ک")
#     text = self.BOUNDARY_SYMBOL_RE.sub(". ", text)
#     text = self.DISALLOWED_PERSIAN_SYMBOL_RE.sub(" ", text)
#     text = self._fix_commas_after_verbs(text)

#     text = re.sub(r"\s+", " ", text)
#     text = re.sub(r"\s+([،.!؟])", r"\1", text)
#     text = re.sub(r"([.!؟]){2,}", r"\1", text)
#     text = re.sub(r"\.\s*\.", ".", text)

#     return text.strip()


# def _persian(self, text: str) -> str:
#     if not text.strip() or not self.FA_RE.search(text):
#         return text

#     text = self._normalize_persian(text)

#     out = self.persian_llm.create_chat_completion(
#         messages=[{"role": "user", "content": f"Transliterate: {text}"}],
#         max_tokens=768,
#         temperature=0.0,
#         repeat_penalty=1.0,
#         stop=["<|im_end|>"],
#     )

#     return out["choices"][0]["message"]["content"]
import re
from dataclasses import dataclass
from typing import Literal, Optional

Mode = Literal["all", "persian", "russian", "tajik"]

@dataclass
class TextPreprocessor:
    mode: Mode = "all"
    persian_llm: Optional[Llama] = None
    phonemizer: Optional[RUPhon] = None
    accentizer: Optional[RUAccent] = None

    TAG_RE = re.compile(r"<[^>\s]+>")
    EMOJI_RE = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    PROTECTED_RE = re.compile(
        r"<[^>\s]+>|[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]+"
    )

    # Persian/Arabic LETTERS only.
    # Important: do not use [\u0600-\u06FF], because that also catches Persian digits.
    # If <tajik> contains only digits + Cyrillic, it must stay in the Tajik path.
    FA_RE = re.compile(
        r"[\u0621-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC]"
    )

    RU_RE = re.compile(r"[А-Яа-яЁё]")

    # Tajik Cyrillic includes standard Cyrillic plus Tajik-specific letters.
    TJ_RE = re.compile(r"[А-Яа-яЁёҒғӢӣҚқӮӯҲҳҶҷ]")

    # Useful for detecting untagged Tajik Cyrillic before Russian.
    TJ_SPECIFIC_RE = re.compile(r"[ҒғӢӣҚқӮӯҲҳҶҷ]")

    TJ_DIGIT_RE = re.compile(r"[0-9۰-۹٠-٩]")

    TJ_MAP = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "g",

        # In your Tajik target style, ғ is q, not gh.
        "ғ": "q",

        "д": "d",
        "ж": "zh",
        "з": "z",
        "и": "i",
        "ӣ": "i",
        "й": "y",
        "к": "k",
        "қ": "q",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "A",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ӯ": "u",
        "ф": "f",
        "х": "kh",
        "ҳ": "h",
        "ч": "ch",
        "ҷ": "j",
        "ш": "sh",
        "ъ": "'",

        # Tajik / loanword letters.
        "э": "e",
        "ц": "ts",
        "щ": "sh",
        "ы": "i",
        "ь": "",
    }

    TJ_WORD_OVERRIDES = {
        # Very common function words / forms matching your target style.
        "ва": "va",
        "на": "na",
        "ҳа": "ha",
        "ба": "ba",
        "аз": "az",
        "дар": "dar",
        "бо": "bA",
        "ки": "ki",
        "ин": "in",
        "он": "An",
        "ман": "man",
        "ту": "tu",
        "ӯ": "u",
        "у": "u",
        "мо": "mA",
        "шумо": "shomA",
        "ҳама": "hame",
        "ҳам": "ham",
        "ҳамин": "hamin",
        "ҳанӯз": "hanuz",

        # Match uploaded Tajik target style: khod, not khud.
        "худ": "khod",
        "худаш": "khodash",
        "худро": "khodrA",
        "худам": "khodam",
        "худат": "khodat",
        "худашон": "khodashAn",

        # Common Persian/Tajik forms in the target style.
        "шуд": "shod",
        "шаванд": "shavan",
        "шуда": "shode",
        "буд": "bud",
        "буда": "bude",
        "нест": "nist",
        "аст": "ast",
        "ҳаст": "hast",

        # Common examples.
        "хона": "khAne",
        "кор": "kAr",
        "рӯз": "ruz",
        "ҷой": "jAy",
        "ҷои": "jAi",
        "роҳ": "rAh",

        # Useful for your Bloomberg test.
        "блумберг": "blumberg",
        "киштӣ": "kishti",
        "киштиро": "kishtirA",
        "киштиҳо": "kishtihA",
        "киштиҳои": "kishtihAi",
        "ҳرمуз": "hurmuz",
        "ҳурмуз": "hurmuz",
        "эро": "erA",
        "эрон": "erAn",
        "эрони": "erAni",
        "эронии": "erAnii",
        "тағйир": "taqyir",
    }

    TJ_DIGIT_WORDS = {
        "0": "sefr",
        "1": "yak",
        "2": "du",
        "3": "se",
        "4": "chohor",
        "5": "panj",
        "6": "shash",
        "7": "haft",
        "8": "hasht",
        "9": "nuh",

        "۰": "sefr",
        "۱": "yak",
        "۲": "du",
        "۳": "se",
        "۴": "chohor",
        "۵": "panj",
        "۶": "shash",
        "۷": "haft",
        "۸": "hasht",
        "۹": "nuh",

        "٠": "sefr",
        "١": "yak",
        "٢": "du",
        "٣": "se",
        "٤": "chohor",
        "٥": "panj",
        "٦": "shash",
        "٧": "haft",
        "٨": "hasht",
        "٩": "nuh",
    }

    @classmethod
    def load(
        cls,
        mode: Mode = "all",
        persian_model_path: str = "/home/ubuntu/zs_cleaning/darya-tts/Text_Preprocessors/Finglish/persian_transliterator-q8_0.gguf",
        ruphon_workdir: str = "./models",
        device: str = "CPU",
        n_ctx: int = 768,
        n_threads: int = 8,
    ):
        self = cls(mode=mode)

        # Tajik mode still needs Persian LLM for this case:
        # <tajik> من به خانه می‌روم.
        # Since that is Perso-Arabic script, it should be treated as Persian.
        if mode in ("all", "persian", "tajik"):
            self.persian_llm = Llama(
                model_path=persian_model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
            )

        if mode in ("all", "russian"):
            self.phonemizer = RUPhon.load("small", workdir=ruphon_workdir, device=device)

            self.accentizer = RUAccent()
            self.accentizer.load(
                omograph_model_size="turbo3",
                use_dictionary=True,
                tiny_mode=False,
            )

        return self

    def _split_protected(self, text: str):
        parts = []
        last = 0

        for m in self.PROTECTED_RE.finditer(text):
            if m.start() > last:
                parts.append(("text", text[last:m.start()]))
            parts.append(("protected", m.group(0)))
            last = m.end()

        if last < len(text):
            parts.append(("text", text[last:]))

        return parts

    def _persian(self, text: str) -> str:
        if not text.strip() or not self.FA_RE.search(text):
            return text

        if self.persian_llm is None:
            raise RuntimeError(
                "Persian LLM is not loaded. Use mode='all', mode='persian', or mode='tajik'."
            )

        out = self.persian_llm.create_chat_completion(
            messages=[{"role": "user", "content": f"Transliterate: {text}"}],
            max_tokens=768,
            min_p=0.05,
            temperature=0.0,
            repeat_penalty=1.0,
            stop=["<|im_end|>"],
        )

        return out["choices"][0]["message"]["content"]

    def _russian(self, text: str) -> str:
        if not text.strip() or not self.RU_RE.search(text):
            return text

        if self.accentizer is None or self.phonemizer is None:
            raise RuntimeError(
                "Russian models are not loaded. Use mode='all' or mode='russian'."
            )

        accented = self.accentizer.process_all(text)
        return self.phonemizer.phonemize(
            accented,
            put_stress=True,
            stress_symbol="^",
        )

    def _tj_iotated(self, ch: str, prev_ch: Optional[str]) -> str:
        """
        Handles е / ё / ю / я.

        For the target style:
        - initial е usually becomes e
        - verb prefix ме- is handled separately as mi-
        - ё / ю / я become yo / yu / ya
        """
        prev_is_boundary = prev_ch is None
        prev_allows_y = prev_ch in {
            "а", "е", "ё", "и", "ӣ", "о", "у", "ӯ", "э", "ю", "я", "й", "ъ"
        }

        if ch == "е":
            return "ye" if not prev_is_boundary and prev_allows_y else "e"
        if ch == "ё":
            return "yo"
        if ch == "ю":
            return "yu"
        if ch == "я":
            return "ya"

        return ch

    def _tajik_number(self, token: str) -> str:
        """
        Converts single digits to the target Tajik romanized words.
        Multi-digit numbers are left unchanged to avoid bad guesses.
        """
        if token in self.TJ_DIGIT_WORDS:
            return self.TJ_DIGIT_WORDS[token]

        return token

    def _tajik_word(self, word: str) -> str:
        """
        Rule-based Tajik Cyrillic word -> your target Tajik romanization style.

        Examples:
        хона -> khAne
        кор -> kAr
        тағйир -> taqyir
        """
        word = word.lower()

        if word in self.TJ_WORD_OVERRIDES:
            return self.TJ_WORD_OVERRIDES[word]

        out = []
        i = 0

        # In the uploaded target style, Tajik verbal ме- is normally mi-:
        # меравад -> miravad
        # мегӯяд / мегуфт style forms -> mi...
        # мехоҳад -> mikhAhad
        if word.startswith("ме") and len(word) > 2:
            out.append("mi")
            i = 2

        while i < len(word):
            ch = word[i]
            prev_ch = word[i - 1] if i > 0 else None

            if ch in {"е", "ё", "ю", "я"}:
                out.append(self._tj_iotated(ch, prev_ch))
                i += 1
                continue

            # Mainland Persian-style final -а often corresponds to -e:
            # хона -> khAne
            # карда -> karde
            # омада -> Amade
            #
            # Common exceptions like ба / ва / на / ҳа are handled above.
            if ch == "а" and i == len(word) - 1 and len(word) > 1:
                out.append("e")
                i += 1
                continue

            out.append(self.TJ_MAP.get(ch, ch))
            i += 1

        return "".join(out)

    def _tajik(self, text: str) -> str:
        """
        Deterministic Tajik Cyrillic -> your target Tajik romanization style.

        No LLM.
        No phonemizer.

        Example:
        Блумберг: 8 киштӣ тағйир доданд.
        blumberg: hasht kishti taqyir dAdand.
        """
        if not text.strip():
            return text

        if not self.TJ_RE.search(text) and not self.TJ_DIGIT_RE.search(text):
            return text

        def repl(m):
            token = m.group(0)

            if self.TJ_DIGIT_RE.fullmatch(token):
                return self._tajik_number(token)

            return self._tajik_word(token)

        return re.sub(
            r"[А-Яа-яЁёҒғӢӣҚқӮӯҲҳҶҷ]+|[0-9۰-۹٠-٩]",
            repl,
            text,
        )

    def _process_text_chunk(self, chunk: str, lang: Optional[str] = None) -> str:
        if not chunk:
            return chunk

        m = re.match(r"^(\s*)(.*?)(\s*)$", chunk, flags=re.DOTALL)
        leading, core, trailing = m.groups()

        if not core:
            return chunk

        if lang == "tajik":
            if self.mode in ("all", "tajik", "persian"):
                if self.FA_RE.search(core):
                    # <tajik> tag, but Perso-Arabic script.
                    # Bypass Tajik logic and treat as normal Persian.
                    core = self._persian(core)

                elif self.TJ_RE.search(core) or self.TJ_DIGIT_RE.search(core):
                    # <tajik> tag with Tajik Cyrillic / digits.
                    # Use deterministic Tajik transliteration.
                    core = self._tajik(core)

        elif lang == "persian":
            if self.mode in ("all", "persian"):
                core = self._persian(core)

        elif lang == "russian":
            if self.mode in ("all", "russian"):
                core = self._russian(core)

        else:
            if self.mode == "tajik":
                if self.FA_RE.search(core):
                    core = self._persian(core)
                elif self.TJ_RE.search(core) or self.TJ_DIGIT_RE.search(core):
                    core = self._tajik(core)

            elif self.mode == "persian":
                core = self._persian(core)

            elif self.mode == "russian":
                core = self._russian(core)

            elif self.mode == "all":
                if self.FA_RE.search(core):
                    core = self._persian(core)

                # Untagged text with Tajik-specific Cyrillic letters should not
                # accidentally go through Russian.
                elif self.TJ_SPECIFIC_RE.search(core):
                    core = self._tajik(core)

                elif self.RU_RE.search(core):
                    core = self._russian(core)

        return leading + core.strip() + trailing

    def process(self, text: str) -> str:
        result = []
        active_lang = None

        for kind, chunk in self._split_protected(text):
            if kind == "protected":
                tag = chunk.lower()

                if tag == "<tajik>":
                    active_lang = "tajik"
                    result.append(chunk)
                    continue

                if tag == "<persian>":
                    active_lang = "persian"
                    result.append(chunk)
                    continue

                if tag in ("<rus>", "<russian>"):
                    active_lang = "russian"
                    result.append(chunk)
                    continue

                # Speaker tags like <S1>, <S2> are preserved and do not reset language.
                result.append(chunk)

            else:
                result.append(self._process_text_chunk(chunk, active_lang))

        return "".join(result)