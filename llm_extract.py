#!/usr/bin/env python3
"""
Tiny LLM for extracting music generation parameters from natural language.
Usage: python llm_extract.py "make it happier, faster, sing about the ocean"
"""

import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

PROMPT_TEMPLATE = """<|system|>
You are a music parameter extractor. Extract BPM, key, and lyrics from user requests.
- BPM: infer from words like "faster" (+20), "slower" (-20), "fast" (140), "slow" (70), or keep current if not mentioned
- Key: infer from "happy", "upbeat", "joy" -> "major", "sad", "dark", "melancholy" -> "minor", or keep current
- Lyrics: extract what to sing about, or "[instrumental]" if nothing mentioned
Output ONLY valid JSON like: {"bpm": 140, "key": "major", "lyrics": "the ocean"}
<|user|>
{input}
<|assistant|>
"""

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        return
    
    user_input = sys.argv[1]
    
    # Load model and tokenizer (cached after first load)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Format prompt
    prompt = PROMPT_TEMPLATE.format(input=user_input)
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.1,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    try:
        # Try to find JSON in response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback: simple keyword extraction
            result = extract_simple(user_input)
    except:
        result = extract_simple(user_input)
    
    print(json.dumps(result))

def extract_simple(text: str) -> dict:
    """Simple keyword-based extraction as fallback."""
    text = text.lower()
    
    bpm = 120  # default
    if "faster" in text or "fast" in text:
        bpm = 140
    elif "slower" in text or "slow" in text:
        bpm = 80
    
    key = "minor"  # default
    if "happy" in text or "upbeat" in text or "joy" in text:
        key = "major"
    
    # Extract lyrics - everything after "sing about" or "about"
    lyrics = "[instrumental]"
    if "sing about" in text:
        idx = text.index("sing about") + len("sing about")
        lyrics = text[idx:].strip().rstrip(".,!?")
    elif "about " in text and len(text) > text.index("about ") + 6:
        idx = text.index("about ") + len("about ")
        lyrics = text[idx:].strip().rstrip(".,!?")
    
    return {"bpm": bpm, "key": key, "lyrics": lyrics}

if __name__ == "__main__":
    main()
