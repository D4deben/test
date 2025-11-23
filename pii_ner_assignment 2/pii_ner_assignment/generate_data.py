"""
Generate synthetic STT-style transcripts with PII entities.
Focus on noisy patterns like spoken numbers, missing punctuation, etc.
"""
import json
import random
from typing import List, Dict, Tuple

random.seed(42)

# Templates for different entity types
CREDIT_CARD_TEMPLATES = [
    "my card number is {cc}",
    "the credit card is {cc}",
    "card ending in {cc}",
    "pay with {cc}",
    "charged to {cc}",
]

PHONE_TEMPLATES = [
    "call me on {phone}",
    "my number is {phone}",
    "reach me at {phone}",
    "phone {phone}",
    "contact {phone}",
]

EMAIL_TEMPLATES = [
    "email me at {email}",
    "my email is {email}",
    "send it to {email}",
    "contact {email}",
    "write to {email}",
]

PERSON_NAME_TEMPLATES = [
    "my name is {name}",
    "this is {name}",
    "i am {name}",
    "call me {name}",
    "{name} speaking",
]

DATE_TEMPLATES = [
    "on {date}",
    "scheduled for {date}",
    "arriving {date}",
    "meeting on {date}",
    "born on {date}",
]

CITY_TEMPLATES = [
    "i live in {city}",
    "located in {city}",
    "from {city}",
    "office in {city}",
    "traveling to {city}",
]

LOCATION_TEMPLATES = [
    "at {location}",
    "near {location}",
    "by {location}",
    "close to {location}",
]

# Sample data pools
FIRST_NAMES = ["john", "sarah", "michael", "emma", "david", "lisa", "robert", "jennifer", 
               "william", "mary", "james", "patricia", "richard", "linda", "thomas", "susan",
               "ramesh", "priya", "amit", "anjali", "raj", "sneha", "vijay", "pooja"]

LAST_NAMES = ["smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis",
              "rodriguez", "martinez", "sharma", "kumar", "patel", "singh", "gupta", "reddy"]

CITIES = ["chennai", "mumbai", "bangalore", "delhi", "hyderabad", "pune", "new york", 
          "london", "paris", "tokyo", "sydney", "toronto", "seattle", "austin"]

LOCATIONS = ["central park", "main street", "fifth avenue", "baker street", "wall street",
             "the mall", "times square", "the airport", "railway station", "city center"]

DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "company", "work", "email"]


def generate_credit_card() -> Tuple[str, str]:
    """Generate spoken credit card number"""
    # Real Luhn-valid test card
    cards = [
        "4242424242424242",  # Visa
        "5555555555554444",  # Mastercard
        "378282246310005",   # Amex
    ]
    cc = random.choice(cards)
    
    # Convert to spoken form
    variants = []
    
    # All digits spoken
    spoken = " ".join(cc)
    variants.append(spoken)
    
    # Groups of 4
    groups = [cc[i:i+4] for i in range(0, len(cc), 4)]
    spoken_groups = " ".join([" ".join(g) for g in groups])
    variants.append(spoken_groups)
    
    # Mixed: some grouped some not
    if len(cc) == 16:
        mixed = f"{cc[:4]} {cc[4:8]} {cc[8:12]} {cc[12:]}"
        variants.append(mixed)
    
    chosen = random.choice(variants)
    return chosen, cc


def generate_phone() -> str:
    """Generate spoken phone number"""
    # Generate 10 digit number
    digits = [str(random.randint(0, 9)) for _ in range(10)]
    
    variants = []
    # All digits spoken
    variants.append(" ".join(digits))
    
    # With words like "oh" for zero, "double" for repeated
    spoken = []
    for d in digits:
        if d == "0":
            spoken.append(random.choice(["zero", "oh"]))
        else:
            spoken.append(d)
    variants.append(" ".join(spoken))
    
    # Grouped (xxx) xxx-xxxx as spoken form
    grouped = f"{''.join(digits[:3])} {''.join(digits[3:6])} {''.join(digits[6:])}"
    variants.append(grouped)
    
    return random.choice(variants)


def generate_email() -> str:
    """Generate spoken email"""
    fname = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)
    domain = random.choice(DOMAINS)
    
    # Convert to spoken form
    variants = [
        f"{fname} dot {lname} at {domain} dot com",
        f"{fname}{lname} at {domain} dot com",
        f"{fname} underscore {lname} at {domain} dot com",
        f"{fname}.{lname}@{domain}.com",  # Some written form
    ]
    return random.choice(variants)


def generate_person_name() -> str:
    """Generate person name"""
    fname = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)
    
    variants = [
        f"{fname} {lname}",
        f"{fname}",
        f"{lname}",
        f"{fname.capitalize()} {lname.capitalize()}",
    ]
    return random.choice(variants)


def generate_date() -> str:
    """Generate spoken date"""
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    month = random.choice(months)
    day = random.randint(1, 28)
    year = random.randint(1990, 2024)
    
    variants = [
        f"{month} {day} {year}",
        f"{day} {month} {year}",
        f"{month} {day}",
        f"{day} {month}",
        f"{month} {year}",
        f"{day} slash {random.randint(1,12)} slash {year}",
        f"{random.randint(1,12)} {day} {year}",
    ]
    return random.choice(variants)


def generate_city() -> str:
    return random.choice(CITIES)


def generate_location() -> str:
    return random.choice(LOCATIONS)


def create_utterance(utt_id: str) -> Dict:
    """Create a single synthetic utterance with entities"""
    entities = []
    text_parts = []
    
    # Decide how many entities to include (1-4)
    num_entities = random.randint(1, 4)
    entity_types = random.sample(
        ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"],
        k=min(num_entities, 7)
    )
    
    # Add some filler text
    fillers = ["", "also", "and", "please", "i need", "can you", ""]
    
    for etype in entity_types:
        # Add filler
        if text_parts:
            text_parts.append(random.choice(fillers))
        
        if etype == "CREDIT_CARD":
            template = random.choice(CREDIT_CARD_TEMPLATES)
            spoken, _ = generate_credit_card()
            text_parts.append(template.replace("{cc}", spoken))
        elif etype == "PHONE":
            template = random.choice(PHONE_TEMPLATES)
            phone = generate_phone()
            text_parts.append(template.replace("{phone}", phone))
        elif etype == "EMAIL":
            template = random.choice(EMAIL_TEMPLATES)
            email = generate_email()
            text_parts.append(template.replace("{email}", email))
        elif etype == "PERSON_NAME":
            template = random.choice(PERSON_NAME_TEMPLATES)
            name = generate_person_name()
            text_parts.append(template.replace("{name}", name))
        elif etype == "DATE":
            template = random.choice(DATE_TEMPLATES)
            date = generate_date()
            text_parts.append(template.replace("{date}", date))
        elif etype == "CITY":
            template = random.choice(CITY_TEMPLATES)
            city = generate_city()
            text_parts.append(template.replace("{city}", city))
        elif etype == "LOCATION":
            template = random.choice(LOCATION_TEMPLATES)
            location = generate_location()
            text_parts.append(template.replace("{location}", location))
    
    # Join and clean
    text = " ".join(text_parts).strip()
    text = " ".join(text.split())  # normalize whitespace
    
    # Now extract entities by finding the generated values
    # This is simplified - in real case would need more careful tracking
    # For now, we'll use a simpler approach: re-scan the text
    
    entities = extract_entities_from_text(text, entity_types)
    
    return {
        "id": utt_id,
        "text": text,
        "entities": entities
    }


def extract_entities_from_text(text: str, expected_types: List[str]) -> List[Dict]:
    """Extract entities by pattern matching the text"""
    import re
    entities = []
    
    # CREDIT_CARD: look for digit patterns
    if "CREDIT_CARD" in expected_types:
        # Match sequences of digits (at least 13)
        for m in re.finditer(r'\b[\d\s]{20,50}\b', text):
            span_text = m.group()
            digit_count = len(re.sub(r'\D', '', span_text))
            if digit_count >= 13:
                entities.append({
                    "start": m.start(),
                    "end": m.end(),
                    "label": "CREDIT_CARD"
                })
    
    # PHONE: digit sequences
    if "PHONE" in expected_types:
        for m in re.finditer(r'\b[\d\s]{10,30}\b', text):
            span_text = m.group()
            digit_count = len(re.sub(r'\D', '', span_text))
            if 7 <= digit_count <= 15 and not any(e["start"] <= m.start() < e["end"] for e in entities):
                entities.append({
                    "start": m.start(),
                    "end": m.end(),
                    "label": "PHONE"
                })
    
    # EMAIL: look for email patterns
    if "EMAIL" in expected_types:
        for m in re.finditer(r'\b\S+\s*@\s*\S+\s*dot\s*\S+\b', text):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "EMAIL"
            })
        # Also look for written form
        for m in re.finditer(r'\b[\w.]+@[\w.]+\b', text):
            entities.append({
                "start": m.start(),
                "end": m.end(),
                "label": "EMAIL"
            })
    
    # PERSON_NAME: look for capitalized or known names
    if "PERSON_NAME" in expected_types:
        for fname in FIRST_NAMES:
            for m in re.finditer(r'\b' + re.escape(fname) + r'\b', text, re.IGNORECASE):
                # Check if followed by last name
                end = m.end()
                for lname in LAST_NAMES:
                    pattern = r'\s+' + re.escape(lname) + r'\b'
                    m2 = re.match(pattern, text[end:], re.IGNORECASE)
                    if m2:
                        entities.append({
                            "start": m.start(),
                            "end": end + m2.end(),
                            "label": "PERSON_NAME"
                        })
                        break
                else:
                    # Just first name
                    if not any(e["start"] <= m.start() < e["end"] for e in entities):
                        entities.append({
                            "start": m.start(),
                            "end": m.end(),
                            "label": "PERSON_NAME"
                        })
    
    # DATE: look for month names or date patterns
    if "DATE" in expected_types:
        months = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november", "december"]
        for month in months:
            pattern = r'\b' + month + r'\s+\d{1,2}(?:\s+\d{4})?\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "start": m.start(),
                    "end": m.end(),
                    "label": "DATE"
                })
    
    # CITY
    if "CITY" in expected_types:
        for city in CITIES:
            pattern = r'\b' + re.escape(city) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                if not any(e["start"] <= m.start() < e["end"] for e in entities):
                    entities.append({
                        "start": m.start(),
                        "end": m.end(),
                        "label": "CITY"
                    })
    
    # LOCATION
    if "LOCATION" in expected_types:
        for loc in LOCATIONS:
            pattern = r'\b' + re.escape(loc) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                if not any(e["start"] <= m.start() < e["end"] for e in entities):
                    entities.append({
                        "start": m.start(),
                        "end": m.end(),
                        "label": "LOCATION"
                    })
    
    # Sort by start position and remove overlaps (keep first)
    entities.sort(key=lambda x: x["start"])
    filtered = []
    for e in entities:
        if not any(f["start"] <= e["start"] < f["end"] or e["start"] <= f["start"] < e["end"] for f in filtered):
            filtered.append(e)
    
    return filtered


def generate_dataset(output_path: str, num_samples: int, start_id: int = 1):
    """Generate synthetic dataset"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            utt_id = f"utt_{start_id + i:04d}"
            utt = create_utterance(utt_id)
            f.write(json.dumps(utt, ensure_ascii=False) + "\n")
    print(f"Generated {num_samples} samples to {output_path}")


if __name__ == "__main__":
    # Generate training data (600 examples)
    generate_dataset("data/train.jsonl", 600, start_id=1)
    
    # Generate dev data (150 examples)
    generate_dataset("data/dev.jsonl", 150, start_id=10001)
    
    # Generate test data (50 examples, without entities for blind test)
    with open("data/test.jsonl", "w", encoding="utf-8") as f:
        for i in range(50):
            utt_id = f"test_{i+1:04d}"
            utt = create_utterance(utt_id)
            # Remove entities for test set
            test_utt = {"id": utt["id"], "text": utt["text"]}
            f.write(json.dumps(test_utt, ensure_ascii=False) + "\n")
    print("Generated 50 test samples to data/test.jsonl")
