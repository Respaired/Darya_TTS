from cotlet_utils import *
import cutlet
import re

katsu = cutlet.Cutlet(ensure_ascii=False)
katsu.use_foreign_spelling = False
katsu.update_mapping('♡', '♡')
katsu.update_mapping('♪', '♪')
katsu.update_mapping('○', '○')
katsu.update_mapping('っ', 'っ')  # Keep っ as is for gemination
katsu.update_mapping('ッ', 'ッ')  # Keep ッ as is for gemination
katsu.update_mapping('◆', '◆')  # Preserve our glottal stop marker

def has_capital_letters(text):
    """Check if text contains capital letters (indicating English words)"""
    return bool(re.search(r'[A-Z]', text))

def split_japanese_english(text):
    """Split text into segments of Japanese and English (with capitals) parts"""
    # Use regex to find all English words with capital letters
    # This pattern matches sequences of letters that contain at least one capital
    pattern = r'[A-Z][A-Za-z]*|[a-z]+[A-Z][A-Za-z]*'
    
    segments = []
    last_end = 0
    
    for match in re.finditer(pattern, text):
        start, end = match.span()
        
        # Add any Japanese text before this English word
        if start > last_end:
            japanese_part = text[last_end:start]
            segments.append((japanese_part, False))
        
        # Add the English word
        english_word = match.group()
        segments.append((english_word, True))
        last_end = end
    
    # Add any remaining Japanese text
    if last_end < len(text):
        japanese_part = text[last_end:]
        segments.append((japanese_part, False))
    
    # If no English words were found, return the whole text as Japanese
    if not segments:
        segments = [(text, False)]
    
    return segments

def process_japanese_text(ml):
    # Check for small characters and replace them
    if any(char in ml for char in "ぁぃぅぇぉ"):
        
        ml = ml.replace("ぁ", "あ")
        ml = ml.replace("ぃ", "い")
        ml = ml.replace("ぅ", "う")
        ml = ml.replace("ぇ", "え")
        ml = ml.replace("ぉ", "お")

    # Initialize Cutlet for romaji conversion

    # Convert to romaji and apply transformations
    # output = katsu.romaji(ml, capitalize=False).lower()

    output = katsu.romaji(apply_transformations(alphabetreading(ml)), capitalize=False).lower()
    

    # Replace specific romaji sequences
    if 'j' in output:
        output = output.replace('j', "dʑ")
    if 'tt' in output:
        output = output.replace('tt', "ʔt")
    if 't t' in output:
        output = output.replace('t t', "ʔt")
    if ' ʔt' in output:
        output = output.replace(' ʔt', "ʔt")
    if 'ssh' in output:
        output = output.replace('ssh', "ɕɕ")

    # Convert romaji to IPA
    output = Roma2IPA(output)

    
    output = hira2ipa(output)

    # Apply additional transformations
    output = replace_chars_2(output)
    output = replace_repeated_chars(replace_tashdid_2(output))
    output = nasal_mapper(output)

    # Final adjustments
    if " ɴ" in output:
        output = output.replace(" ɴ", "ɴ")
        
    if ' neɽitai ' in output:
        output = output.replace(' neɽitai ', "naɽitai")

    if 'harɯdʑisama' in output:
        output = output.replace('harɯdʑisama', "arɯdʑisama")


    if "ki ni ɕinai" in output:
        output = re.sub(r'(?<!\s)ki ni ɕinai', r' ki ni ɕinai', output)

    if 'ʔt' in output:
        output = re.sub(r'(?<!\s)ʔt', r'ʔt', output)

    if 'de aɽoɯ' in output:
        output = re.sub(r'(?<!\s)de aɽoɯ', r' de aɽoɯ', output)

        
    return output.lstrip()

# def replace_repeating_patterns(text):
#     def replace_repeats(match):
#         pattern = match.group(1)
#         if len(match.group(0)) // len(pattern) >= 3:
#             return pattern + "~~~"
#         return match.group(0)

#     # Pattern for space-separated repeats
#     pattern1 = r'((?:\S+\s+){1,5}?)(?:\1){2,}'
#     # Pattern for continuous repeats without spaces
#     pattern2 = r'(.+?)\1{2,}'

#     text = re.sub(pattern1, replace_repeats, text)
#     text = re.sub(pattern2, replace_repeats, text)
#     return text


def replace_repeating_a(output):
    # Define patterns and their replacements
    patterns = [
        (r'(aː)\s*\1+\s*', r'\1~'),  # Replace repeating "aː" with "aː~~"
        (r'(aːa)\s*aː', r'\1~'),     # Replace "aːa aː" with "aː~~"
        (r'aːa', r'aː~'),             # Replace "aːa" with "aː~"
        (r'naː\s*aː', r'naː~'),       # Replace "naː aː" with "naː~"
        (r'(oː)\s*\1+\s*', r'\1~'),  # Replace repeating "oː" with "oː~~"
        (r'(oːo)\s*oː', r'\1~'),     # Replace "oːo oː" with "oː~~"
        (r'oːo', r'oː~'),              # Replace "oːo" with "oː~"
        (r'(eː)\s*\1+\s*', r'\1~'),  
        (r'(e)\s*\1+\s*', r'\1~'),  
        (r'(eːe)\s*eː', r'\1~'),     
        (r'eːe', r'eː~'),             
        (r'neː\s*eː', r'neː~'),       
    ]

    
    # Apply each pattern to the output
    for pattern, replacement in patterns:
        output = re.sub(pattern, replacement, output)
    
    return output

def merge_katakana_phonemes(output, original_text):
    """Merge phonemes that likely came from katakana words"""
    # Check if text is mostly katakana (if so, don't merge)
    text_without_punct = re.sub(r'[!?！？。、,.\s]', '', original_text)
    katakana_chars = len(re.findall(r'[\u30A0-\u30FF]', text_without_punct))
    total_chars = len(text_without_punct)
    
    if total_chars == 0 or (katakana_chars / total_chars > 0.8):
        # Entire text is katakana or empty, keep normal spacing
        return output
    
    # Check if original text has any katakana
    if not re.search(r'[\u30A0-\u30FF]', original_text):
        # No katakana in original text, no need to merge
        return output
    
    # Pattern to find potential katakana-originated phoneme sequences
    # These typically have long vowels (ː) and are short syllables
    # Common pattern: syllable with long vowel followed by another syllable
    pattern = r'\b([a-z]{1,3}ː)\s+([a-z]{1,3}(?:ː)?[a-z]{0,2})\b'
    
    def should_merge(match):
        """Determine if two phoneme parts should be merged"""
        part1, part2 = match.groups()
        # Merge if both parts are short and at least one has a long vowel
        if len(part1) <= 4 and len(part2) <= 4 and 'ː' in (part1 + part2):
            return True
        return False
    
    # Replace matches where merging is appropriate
    result = output
    for match in re.finditer(pattern, result):
        if should_merge(match):
            original_match = match.group(0)
            merged = match.group(1) + match.group(2)
            result = result.replace(original_match, merged, 1)
    
    return result

def is_english_text(text):
    """Check if text is English (contains only ASCII letters, numbers, and basic punctuation)"""
    # Remove spaces and basic punctuation to check the core content
    core_text = re.sub(r'[\s\.,!?\-\'\"]+', '', text)
    if not core_text:
        return False
    
    # Check if it's primarily ASCII letters and numbers (English)
    # Allow some common English punctuation within the text
    english_chars = len(re.findall(r'[A-Za-z0-9]', core_text))
    total_chars = len(core_text)
    
    # If more than 80% of characters are English letters/numbers, treat as English
    if total_chars > 0 and english_chars / total_chars > 0.8:
        return True
    
    # Also check if it contains any capital letters (original logic for proper nouns)
    if re.search(r'[A-Z]', text):
        return True
    
    return False

def split_japanese_english(text):
    """Split text into segments of Japanese and English parts"""
    segments = []
    current_segment = ""
    current_is_english = None
    
    # Process character by character to better handle mixed text
    i = 0
    while i < len(text):
        # Look ahead to get a word or character group
        # Try to match English words first
        english_word_match = re.match(r'[A-Za-z]+[\'\-]?[A-Za-z]*', text[i:])
        
        if english_word_match:
            word = english_word_match.group()
            
            # If we were building a Japanese segment, save it first
            if current_is_english is False and current_segment:
                segments.append((current_segment, False))
                current_segment = ""
            
            current_is_english = True
            current_segment += word
            i += len(word)
            
            # Check if there's whitespace or punctuation after the English word
            while i < len(text) and text[i] in ' \t\n.,!?;:':
                current_segment += text[i]
                i += 1
                
                # If we hit Japanese punctuation, switch context
                if text[i-1] in '。、！？':
                    segments.append((current_segment, True))
                    current_segment = ""
                    current_is_english = None
                    
        else:
            # Not an English word, treat as Japanese or other character
            char = text[i]
            
            # Check if it's a Japanese character or punctuation
            if (ord(char) > 127 or char in '。、！？「」『』（）'):  # Non-ASCII or Japanese punctuation
                if current_is_english is True and current_segment:
                    # Save the English segment
                    segments.append((current_segment.rstrip(), True))
                    current_segment = ""
                
                current_is_english = False
                current_segment += char
            else:
                # ASCII character that's not part of an English word
                if current_segment:
                    current_segment += char
                else:
                    # Start a new segment, determine type based on context
                    current_segment = char
                    current_is_english = False
            
            i += 1
    
    # Add any remaining segment
    if current_segment:
        segments.append((current_segment, current_is_english if current_is_english is not None else False))
    
    # If no segments were created, return the whole text as Japanese
    if not segments:
        segments = [(text, False)]
    
    # Merge consecutive segments of the same type
    merged_segments = []
    for segment, is_eng in segments:
        if merged_segments and merged_segments[-1][1] == is_eng:
            # Merge with previous segment
            merged_segments[-1] = (merged_segments[-1][0] + segment, is_eng)
        else:
            merged_segments.append((segment, is_eng))
    
    return merged_segments

def phonemize(text):
    # Split text into Japanese and English segments
    segments = split_japanese_english(text)
    processed_segments = []
    
    for segment_text, is_english in segments:
        # Check if this segment should be treated as English
        if is_english or is_english_text(segment_text):
            # English text - preserve as-is without IPA conversion
            processed_segments.append(f"[{segment_text.strip()}]")
        else:
            # Store original for katakana checking
            original_segment = segment_text
            
            # Pre-process っ and ッ in specific contexts where they should be a glottal stop
            # Replace っ/ッ when it's:
            # 1. At the end of a word/segment
            # 2. Before a vowel 
            # 3. Before punctuation
            segment_text = re.sub(r'[っッ](?=[あいうえおアイウエオ、。！？…\s]|$)', '◆', segment_text)
            
            # Japanese text - process normally
            segment_text = segment_text.replace("○","|")
            
            output = post_fix(process_japanese_text(segment_text))
            
            if " ɴ" in output:
                output = output.replace(" ɴ", "ɴ")
            if "y" in output:
                output = output.replace("y", "j")
            if "ɯa" in output:
                output = output.replace("ɯa", "wa")
                
            if "a aː" in output:
                output = output.replace("a aː","a~")
            if "a a" in output:
                output = output.replace("a a","a~")

            if "niʔpoɴ go" in output:
                output = output.replace("niʔpoɴ go","nihoɴ go")
                
            output = replace_repeating_a((output))
            output = re.sub(r'\s+~', '~', output)
            
            if "oː~o oː~ o" in output:
                output = output.replace("oː~o oː~ o","oː~~~~~~")
            if "aː~aː" in output:
                output = output.replace("aː~aː","aː~~~")
            if "oɴ naː" in output:
                output = output.replace("oɴ naː","onnaː")
            if "aː~~ aː" in output:
                output = output.replace("aː~~ aː","aː~~~~")
            if "oː~o" in output:
                output = output.replace("oː~o","oː~~")
            if "oː~~o o" in output:
                output = output.replace("oː~~o o","oː~~~~")

            output = random_space_fix(output)
            output = random_sym_fix(output)
            output = random_sym_fix_no_space(output)
            
            if "があ" in original_segment:
                output = output.replace("gaː","ga a")
                
            if "ならあ" in original_segment:
                output = output.replace("naɽaːr","naɽa ar").replace("naɽaːɽ","naɽa aɽ")
                
            if "はア" in original_segment:
                output = output.replace("waː","wa a")
                
            if "がア" in original_segment:
                output = output.replace("gaː","ga a")
            
            output = output.lstrip().replace("... ","...").replace("|","○")
            
            # Fix は pronunciation: should be "ha" at start or after punctuation, not "wa"
            # Pattern: wa at the beginning of the output
            if output.startswith('wa'):
                if original_segment.startswith('は'):
                    output = 'ha' + output[2:]
            
            # Pattern: wa after punctuation/spaces (not functioning as particle)
            output = re.sub(r'([、。！？…,!?\s]+)wa\b', lambda m: m.group(1) + 'ha' 
                          if re.search(r'[、。！？…,!?\s]+は', original_segment) else m.group(0), 
                          output)
            
            # Replace our glottal stop marker with ʔ
            if '◆' in output:
                output = output.replace('◆', 'ʔ')
            
            # Also check if any っ or ッ survived (shouldn't happen but just in case)
            if 'っ' in output:
                output = output.replace('っ', 'ʔ')
            if 'ッ' in output:
                output = output.replace('ッ', 'ʔ')
            
            # Fix any ʔ♡ patterns
            output = output.replace("ʔ♡","♡♡")
            
            # Apply katakana phoneme merging
            output = merge_katakana_phonemes(output, original_segment)
            
            processed_segments.append(output.strip())
    
    # Join all segments back together with spaces
    result = ' '.join(processed_segments)
    
    # Clean up any double spaces
    result = re.sub(r'\s+', ' ', result)
    
    return result.replace(",ɴ",", ɴ").replace("itɕi taikaɴ","iʔtaikaɴ")