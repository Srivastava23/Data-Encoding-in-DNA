import re

with open('website/script.js', 'r', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r'(\"64\":\s*\"#[a-f0-9]{6}\",\s*\"128\":\s*\"#[a-f0-9]{6}\",\s*\"256\":\s*\"#[a-f0-9]{6}\",?)')

def replacer(match):
    original = match.group(1)
    # If it ends with a comma, we just append our new entries
    if original.endswith(','):
        return original + '\n                      "512": "#ec4899",\n                      "1024": "#10b981",'
    else:
        return original + ',\n                      "512": "#ec4899",\n                      "1024": "#10b981"'

new_text = pattern.sub(replacer, text)

with open('website/script.js', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Updated script.js with new colors!")
