# Final comprehensive evaluation script for prosodic constraints (sullable counts and rhyme schemes)
# Evaluation is done across all saved iterations of the finetuning experiments to see how model performance evolves
import os
import json
import torch
import math
import csv
import re
import itertools
import random
import numpy as np
import syllables
import nltk
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pronouncing

testDataPath = "dataset_splits_top15/test.jsonl"
syllableOutputFile = "eval_test/metric_syllable_mse.csv"
rhymeOutputFile = "eval_test/metric_rhyme_scheme.csv"
generationsOutputFile = "eval_test/model_generations.txt"
numSamples = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
maxNewTokens = 80
models = {
    "0_Baseline_GPT2": "gpt2", 
    "1_Gutenberg_Corpus": "/home/ubuntu/mockingbird_transfer_s3/recovery_tokenizer",
    "2_Lyrics_LM_Dataset": "/home/ubuntu/mockingbird_transfer_s3/mockingbird_model_epoch7_l40_epoch1",
    "3_LR_Sweep_2e-05": "results_lr_sweep/lr_2e-05/model",
    "4_LR_Sweep_4e-05": "results_lr_sweep/lr_4e-05/model",
    "5_LR_Sweep_6e-05": "results_lr_sweep/lr_6e-05/model",
    "6_LR_Sweep_1e-4": "results_lr_sweep/lr_0.0001/model", 
    "4_Dropout_0.3": "results_dropout_sweep/lr_0.0001_dr_0.3/model",
    "4_Dropout_0.2": "results_dropout_sweep/lr_0.0001_dr_0.2/model",
    "4_Dropout_0.1": "results_dropout_sweep/lr_0.0001_dr_0.1/model",
    "5_Final_Run_Epoch_3": "results_final_run/best_model" 
}


def lineSquaredError(gtLine, genLine):
    return (syllables.estimate(genLine) - syllables.estimate(gtLine)) ** 2


def calculateMseOfSyllableCounts(gtText, genText):
    gtLines = [l for l in gtText.split("\n") if l.strip()]
    genLines = [l for l in genText.split("\n") if l.strip()]
    maxLength = max(len(gtLines), len(genLines))
    if maxLength == 0:
        return 0.0
    gtLines += [""] * (maxLength - len(gtLines))
    genLines += [""] * (maxLength - len(genLines))
    errors = [lineSquaredError(gt, gen) for gt, gen in zip(gtLines, genLines)]
    return sum(errors) / len(errors)


def cleanStanza(rawOutput, expectedLines=4):
    lines = [l.strip() for l in rawOutput.split("\n") if l.strip()]
    lines = lines[:expectedLines]
    while len(lines) < expectedLines:
        lines.append("")
    return "\n".join(lines)


def getLastWord(text):
    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return None
    return tokens[-1].lower()


def checkWordsRhyme(word1, word2):
    if word1 == word2: 
        return True 
    phones1List = pronouncing.phones_for_word(word1)
    phones2List = pronouncing.phones_for_word(word2)
    if not phones1List or not phones2List:
        return False
    for p1 in phones1List:
        for p2 in phones2List:
            if pronouncing.rhyming_part(p1) == pronouncing.rhyming_part(p2):
                return True
    return False


def extractRhymeSchemePattern(stanzaText):
    lines = [l for l in stanzaText.split('\n') if l.strip()]
    lastWords = [getLastWord(l) for l in lines]
    schemeIds = [-1] * len(lastWords)
    nextId = 0
    for i in range(len(lastWords)):
        if schemeIds[i] != -1:
            continue
        currentWord = lastWords[i]
        schemeIds[i] = nextId
        for j in range(i + 1, len(lastWords)):
            if schemeIds[j] == -1:
                compareWord = lastWords[j]
                if compareWord and checkWordsRhyme(currentWord, compareWord):
                    schemeIds[j] = nextId
        nextId += 1
    return tuple(schemeIds)


def calculateRhymeSchemeMatch(originalStanza, generatedStanza):
    inputPattern = extractRhymeSchemePattern(originalStanza)
    outputPattern = extractRhymeSchemePattern(generatedStanza)
    if len(inputPattern) != len(outputPattern):
        return 0.0
    return 1.0 if inputPattern == outputPattern else 0.0


def loadData(path, limit=None):
    inputs = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            fullText = data['text']
            lines = fullText.strip().split('\n')
            promptLines = lines[:4]
            prompt = "\n".join(promptLines) + "\n"
            inputs.append(prompt)
    return inputs


def evaluateModel(modelName, modelPath, testInputs, generationTracker):
    tokenizer = GPT2TokenizerFast.from_pretrained(modelPath)
    model = GPT2LMHeadModel.from_pretrained(modelPath)
    model.to(device)
    model.eval()
    
    # Set pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mseScores = []
    rhymeScores = []
    for i, prompt in enumerate(testInputs):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=maxNewTokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            fullText = tokenizer.decode(out[0], skip_special_tokens=True)
            generatedPart = fullText[len(prompt):].strip()
            cleanedOutput = cleanStanza(generatedPart, expected_lines=4)
            cleanPrompt = prompt.strip()
            mse = calculateMseOfSyllableCounts(cleanPrompt, cleanedOutput)
            rhymeMatch = calculateRhymeSchemeMatch(cleanPrompt, cleanedOutput)
            generationTracker[i][modelName] = {
                "text": cleanedOutput,
                "mse": mse,
                "rhyme": rhymeMatch
            }
            mseScores.append(mse)
            rhymeScores.append(rhymeMatch)

    avgMse = sum(mseScores) / len(mseScores) if mseScores else 0.0
    avgRhyme = sum(rhymeScores) / len(rhymeScores) if rhymeScores else 0.0
    return avgMse, avgRhyme


def saveAggregatedGenerations(testInputs, generationResults, modelNames):
    with open(generationsOutputFile, "w") as f:
        f.write("=== GENERATED STANZAS LOG ===\n\n")
        for i, prompt in enumerate(testInputs):
            f.write("input\n")
            for line in prompt.strip().split('\n'):
                f.write(f"{line.strip()}\n")
            f.write("\n")
            for modelName in modelNames:
                resultData = generationResults[i].get(modelName)
                f.write(f"{modelName} output\n")
                if resultData:
                    outputText = resultData["text"]
                    mseScore = resultData["mse"]
                    rhymeScore = resultData["rhyme"]
                    for line in outputText.strip().split('\n'):
                        f.write(f"{line.strip()}\n")
                    f.write(f"Rhyme score {rhymeScore:.4f}        Syllable mse {mseScore:.4f}\n")
                else:
                    f.write("N/A\n")
                f.write("\n")
            f.write("-" * 30 + "\n\n")


def main():
    print("Starting comprehensive evaluation...")
    testInputs = loadData(testDataPath, limit=numSamples)
    generationResults = {i: {} for i in range(len(testInputs))}
    resultsMse = []
    resultsRhyme = []
    for modelName, modelPath in models.items():
        avgMse, avgRhyme = evaluateModel(modelName, modelPath, testInputs, generationResults)
        resultsMse.append([modelName, avgMse])
        resultsRhyme.append([modelName, avgRhyme])
        print(f"Result {modelName}: MSE={avgMse:.4f}, RhymeMatch={avgRhyme:.4f}")
    saveAggregatedGenerations(testInputs, generationResults, models.keys())
    with open(syllableOutputFile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Iteration", "Average_Syllable_MSE"])
        writer.writerows(resultsMse)
    with open(rhymeOutputFile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Iteration", "Rhyme_Scheme_Match_Rate"])
        writer.writerows(resultsRhyme)
    print(f"\nEvaluation Complete. Results saved.")


if __name__ == "__main__":
    main()
