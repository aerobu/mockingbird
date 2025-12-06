# Convert dataset format to "text": <input stanza + \n + parody stanza + \n>
import json
from pathlib import Path

stanzas1 = "stanzas_first_half.txt"
stanzas2 = "stanzas_second_half.txt"
parodies1 = "parodies_llama31_parallel_first_half_all_variants.jsonl"
parodies2 = "parodies_llama31_parallel_second_half_all_variations.jsonl"
output = "dataset_causal_lm.jsonl"

def loadStanzas(path):
    stanzaMap = {}
    with open(path, "r") as f:
        content = f.read().strip()
    blocks = content.split("Stanza ")[1:]
    for block in blocks:
        lines = block.strip().split("\n")
        stanzaId = int(lines[0].strip())
        stanzaText = "\n".join(lines[1:]).strip()
        stanzaMap[stanzaId] = stanzaText
    return stanzaMap

stanzas = {}
stanzas.update(loadStanzas(stanzas1))
stanzas.update(loadStanzas(stanzas2))

def loadParodies(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj)
    return out

parodies = loadParodies(parodies1) + loadParodies(parodies2)

with open(output, "w") as out:
    for entry in parodies:
        stanzaId = entry["stanza_id"]
        stanza = stanzas.get(stanzaId)
        parody = entry["parody"].strip()
        combined = stanza.strip() + "\n" + parody + "\n"
        json.dump({"text": combined}, out)
        out.write("\n")
