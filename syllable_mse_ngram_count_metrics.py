# This code generates parody stanzas using a language model and evaluates them based on syllable count MSE and unique n-gram percentage.

import nltk
from nltk.util import ngrams
from transformers import AutoModelForCausalLM, AutoTokenizer

def line_squared_error(groundtruth_line, generated_line):
  groundtruth_syllables = syllables.estimate(groundtruth_line)
  generated_syllables = syllables.estimate(generated_line)

  se = (generated_syllables - groundtruth_syllables) ** 2

  return se

def calculate_mse_of_syllable_counts(groundtruth_text, generated_text, delimiter='\n'):
  # separate into lines
  groundtruth_lines = groundtruth_text.split(delimiter)
  generated_lines = generated_text.split(delimiter)

  # pad lines if needed with empty strings
  max_lines = max(len(groundtruth_lines), len(generated_lines))
  groundtruth_lines += [''] * (max_lines - len(groundtruth_lines))
  generated_lines += [''] * (max_lines - len(generated_lines))

  mse_per_line = [line_squared_error(gt, gen) for gt, gen in zip(groundtruth_lines, generated_lines)]
  mse = sum(mse_per_line) / len(mse_per_line)

  return mse

def calculate_unique_ngram_percentage(text, n=2):
  tokens = nltk.word_tokenize(text.lower())
  generated_ngrams = list(ngrams(tokens, n))
  unique_ngrams = set(generated_ngrams)
  # print("Generated N-grams:", generated_ngrams)
  # print("Unique N-grams:", unique_ngrams)
  unique_ngram_percentage = len(unique_ngrams) / len(generated_ngrams)

  return unique_ngram_percentage

qbfqbf = "The quick brown fox jumps over the lazy dog. The quick brown fox."
qbfldotd = "The quick brown fox jumps over the lazy dog lazy. dog"
qbfld = "The quick brown fox jumps over the lazy dog lazy dog"

print(calculate_unique_ngram_percentage(qbfqbf))
print(calculate_unique_ngram_percentage(qbfldotd))
print(calculate_unique_ngram_percentage(qbfld))



model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)

messages = [
    {"role": "system", "content": "Always respond in songs, in one stanza that matches rhyme scheme"},
    # {"role": "user", "content": "What is the weather like in Madrid?"},
    {"role": "user", "content": "Create a parody song of this stanza, but make it about Mathematics: \n \n Original song: Fly me to the moon \n Let me play among the stars \n And let me see what spring is like \n On a-Jupiter and Mars \n  \n Parody song:"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="low",
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))

with open("stanzas_test.txt", 'r') as f:
  stanzas_text = f.read()

stanzas = []
next_stanza = ""

for line in stanzas_text.split('\n'):
  if line.startswith('Stanza'):
    if not (next_stanza == ""):
      stanzas.append(next_stanza)
      next_stanza = ""
  elif next_stanza == "":
    next_stanza += line # avoid newlines at beginnings of stanzas
  else:
    next_stanza += ' \n ' + line

stanzas.append(next_stanza) # last stanza

# print(stanzas[:2])

themes = ['Mathematics', 'Soccer', 'Dogs', 'Coding', 'Dinosaurs', 'Christmas', 'Tennis', 'Cars', 'Space', 'Geography', 'Psychology', 'Alphabet', 'Cats', 'NFL', 'Birds', 'Zoo animals', 'Ancient Egypt', 'Monetary Policy', 'Lord of the Rings', 'Dungeons and Dragons', 'Physics', 'Fashion', 'Golf', 'Pokemon', 'Romans', 'Philosophy', 'Counting', 'Sea life & fish', 'Economy', 'Star Wars', 'Taxes']

def truncate_stanza(stanza, lines_in_original_stanza, delimiter='\n'):
  lines = stanza.split(delimiter)
  truncated_lines = lines[:lines_in_original_stanza]
  return delimiter.join(truncated_lines)

def filter_empty_lines(stanza, delimiter='\n'):
  lines = stanza.split(delimiter)
  filtered_lines = [line for line in lines if line.strip() != '']
  return delimiter.join(filtered_lines)

messages = [
    {"role": "system", "content": "Always respond in songs"},
]

NUM_TEXTS_TO_GENERATE_PER_INPUT = 1
NUM_TEXTS_TO_GENERATE = 1 # len(stanzas) * len(themes)
for i in range(NUM_TEXTS_TO_GENERATE):
  original_stanza = stanzas[i]
  lines_in_original_stanza = len(original_stanza.split('\n')) - 1 # newline at end
  theme = themes[i+1 % len(themes)]
  prompt = "Create a parody song of this stanza that rhymes, but make it about " + theme + ". \n\n Original song: \n " + original_stanza + " \n  \n parody:"
  print(prompt)
  messages.append({"role": "user", "content": prompt})

  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt",
      return_dict=True,
  ).to(model.device)

  generated = model.generate(**inputs, max_new_tokens=75)
  print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))

  # generated_texts = generator(prompt, max_new_tokens=75, num_return_sequences=NUM_TEXTS_TO_GENERATE_PER_INPUT)
  # generated_texts = generator("Create a parody of this stanza for me with the theme of mathematics. \n\n Original song: \n fly me to the moon \n and let me play among the stars \n let me see what spring is like \n on Jupiter and mars \n  \n parody:", max_new_tokens=75, num_return_sequences=NUM_TEXTS_TO_GENERATE_PER_INPUT)
  best_parody_text = ""
  best_parody_score = None
  for j in range(len(generated_texts)):
    # print(generated_texts[i])
    gt = generated_texts[j]['generated_text']
    # print(gt)
    parody_text = gt.split('parody:')[1]
    filtered_parody_text = filter_empty_lines(parody_text)
    truncated_original_stanza = truncate_stanza(original_stanza, lines_in_original_stanza)
    truncated_parody_text = truncate_stanza(filtered_parody_text, lines_in_original_stanza)
    # print(parody_text)
    parody_score = -calculate_mse_of_syllable_counts(truncated_original_stanza, truncated_parody_text)
    if best_parody_score == None or best_parody_score < parody_score:
      best_parody_score = parody_score
      best_parody_text = truncated_parody_text
  print(best_parody_text)
  print("Score: " + str(best_parody_score))
  # print("Max score: " + str(-calculate_mse_of_syllable_counts(original_stanza, original_stanza)))

best_parody_text

calculate_mse_of_syllable_counts(original_stanza, best_parody_text)
