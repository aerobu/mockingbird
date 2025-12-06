import random
import string
import pronouncing
import re
import itertools
import numpy as np


def consonant_clusters():
    return ['F W', 'F R', 'F L', 'S W', 'S V',
            'S R', 'S L', 'S N', 'S M', 'S F',
            'S P', 'S T', 'S K', 'SH W', 'SH R',
            'SH L', 'SH N', 'SH M', 'TH W', 'TH R',
            'V W', 'V R', 'V L', 'Z W', 'Z L',
            'B W', 'B R', 'B L', 'D W', 'D R',
            'G W', 'G R', 'G L', 'P W', 'P R',
            'P L', 'T W', 'T R', 'K W', 'K R',
            'K L', 'L Y', 'N Y', 'M Y', 'V Y',
            'H Y', 'F Y', 'S Y', 'TH Y', 'Z Y',
            'B Y', 'D Y', 'G Y', 'P Y', 'T Y',
            'K Y', 'S P L', 'S P R', 'S T R',  'S K R',
            'S K W']


def check_if_consonant_cluster(phones):
    """Return True if CMUdict phonemes is a consonant cluster."""
    return phones in consonant_clusters()


def check_if_vowel(phone):
    """Returns True if CMUdict phoneme is a vowel."""
    # all vowels in CMU Pronouncing Dictionary have stress number 0-2
    return phone[-1] in '012'


def check_if_stressed_vowel(phone):
    """Returns True if CMUdict phoneme is a stressed vowel."""
    # 1 or 2 indicate vowel is stressed
    return phone[-1] in '12'


def check_if_non_stressed_vowel(phone):
    """Returns True if CMUdict phoneme is a non-stressed vowel."""
    # 0 indicates vowel is unstressed
    return phone[-1] in '0'


def check_if_consonant(phone):
    """Returns True if CMUdict phoneme is a consonant."""
    # consonants do not have any stress number
    return phone[-1] not in '012'

def unique(data_list):
    """Removes duplicates from a list, i.e. just unique elements."""
    return list(dict.fromkeys(data_list))


def all_the_same(data_list, val):
    """Returns true if all elements in data_list at equal to val"""
    if len(data_list) > 0 and len(data_list) == data_list.count(val):
        return True
    else:
        return False

def random_phones_for_word(word):
    """Chooses random set of CMUdict phonemes for word

    :param word: a word
    :return: CMUdict phonemes string
    """
    all_phones = pronouncing.phones_for_word(word)
    if not all_phones:
        return ""
    phones = random.choice(all_phones)
    return phones

def first_phones_for_word(word):
    """Chooses first set of CMUdict phonemes for word

    :param word: a word
    :return: CMUdict phonemes string
    """
    all_phones = pronouncing.phones_for_word(word)
    if not all_phones:
        return ""
    phones = all_phones[0]
    return phones

def rhyme(word, phones=None):
    """ Returns a list of rhymes for a word.

    The conditions for this 'normal' rhyme between words are:
    (1) last stressed vowel and subsequent phonemes match
    If phones argument not given, phones/pronunciation used will default to the
    first in the list of phones returned for word. If no rhyme is found, an
    empty list is returned.

    This is the 'default' rhyme, same definition used by the pronoucning
    module for its 'rhymes' function. This is also like the shared set of
    perfect and identical rhymes, except the identical word will be removed
    from the returned rhymes list.


    :param word: a word
    :param phones: specific CMUdict phonemes string for word (default None)
    :return: a rhyme for word
    """

    if phones is None:
        phones = first_phones_for_word(word)
        if phones == "":
            return []
    else:
        if phones not in pronouncing.phones_for_word(word):
            raise ValueError(phones + " not phones for " + word)
    if not phones:
        raise ValueError("phonemes string is empty")
    return [
        w for w in
        pronouncing.rhyme_lookup.get(pronouncing.rhyming_part(phones), [])
        if (w != word)]

def perfect_rhyme(word, phones=None):
    """ Returns a list of perfect rhymes for a word.

    The conditions for a perfect rhyme between words are:
    (1) last stressed vowel and subsequent phonemes match
    (2) onset of last stressed syllable is different
    If phones argument not given, phones/pronunciation used will default to the
    first in the list of phones returned for word. If no rhyme is found, an
    empty list is returned.


    :param word: a word
    :param phones: specific CMUdict phonemes string for word (default None)
    :return: a list of perfect rhymes for word
    """
    if phones is None:
        phones = first_phones_for_word(word)
        if phones == "":
            return []
    else:
        if phones not in pronouncing.phones_for_word(word):
            raise ValueError(phones + " not phones for +" + word)
    if not phones:
        raise ValueError("phonemes string is empty")
    perf_and_iden_rhymes = rhyme(word, phones)
    identical_rhymes = identical_rhyme(word, phones)
    perfect_rhymes = list(np.setdiff1d(perf_and_iden_rhymes, identical_rhymes))
    if word in perfect_rhymes:
        perfect_rhymes.remove(word)
    return perfect_rhymes


def identical_rhyme(word, phones=None):
    """ Returns identical rhymes of word.

    The conditions for an identical rhyme between words are:
    (1) last stressed vowel and subsequent phonemes match
    (2) onset of last stressed syllable is the same
        e.g. 'leave' and 'leave', or 'leave' and 'believe'
    If phones argument not given, phones/pronunciation used will default to the
    first in the list of phones returned for word. If no rhyme is found, an
    empty list is returned.

    The identical part of the word doesn't have to be a 'real' word.
    e.g. The phonemes for 'vection' will be used to find identical rhymes
    of 'convection' (e.g. advection) even though 'vection' is unusual/obscure.


    :param word: a word
    :param phones: specific CMUdict phonemes string for word (default None)
    :return: a list of identical rhymes for word
    """
    if phones is None:
        phones = first_phones_for_word(word)
        if phones == "":
            return []
    else:
        if phones not in pronouncing.phones_for_word(word):
            raise ValueError(phones + " not phones for +" + word)
    if not phones:
        raise ValueError("phonemes string is empty")

    phones_list = phones.split()
    search_list = []
    for i in range(len(phones_list)-1, -1, -1):
        phone = phones_list[i]
        if check_if_stressed_vowel(phone) is False:
            search_list.append(phone)
        else:
            search_list.append(phone)
            last_stressed_vowel_at_start = (i == 0)
            if last_stressed_vowel_at_start is True:
                search_list.reverse()
                search = ' '.join(search_list)
                rhymes = pronouncing.search(search + "$")
                return rhymes
            else:
                consonant_cnt = 0
                consonants= ""
                search_start = ""
                for j in range(i, 0, -1):
                    next_phone = phones_list[j-1]
                    if check_if_consonant(next_phone) is True:
                        consonant_cnt += 1
                        if consonant_cnt > 1:
                            consonants = next_phone + " " + consonants
                            if check_if_consonant_cluster(consonants):
                                search_list.append(next_phone)
                            else:
                                break
                        else:
                            consonants = next_phone
                            search_list.append(next_phone)
                    else:
                        if consonant_cnt == 0:  # null onset
                            # Regex: vowel (AA1, EH0, ect.) or start '^'
                            # pretty sure all vowel start two letters...
                            #   (would be "((.{1,2}(0|1|2) )|^)" otherwise)
                            search_start = "((..(0|1|2) )|^)"
                        break
                search_list.reverse()
                search = search_start + ' '.join(search_list) + "$"
                rhymes = pronouncing.search(search)
                rhymes = unique(rhymes)
                # for r in rhymes:
                #     print(pronouncing.phones_for_word(r)[0])
                return rhymes

def near_rhyme(word, phones=None, stress=True, consonant_tail=0):
    """ Returns a list of words that almost rhyme

    The conditions for a near rhyme between words are:
    (1) At least one of the phonemes after and including the last stressed
        syllable match, except for the case where they all do.
    If phones argument not given, phones/pronunciation used will default to the
    first in the list of phones returned for word. If no rhyme is found, an
    empty list is returned.


    :param word: a word
    :param phones: specific CMUdict phonemes string for word (default None)
    :param stress: if vowels will match stress (default True)
    :param consannt_tail: number of
    :return: a list of near rhymes for word
    """
    if phones is None:
        phones = first_phones_for_word(word)
        if phones == "":
            return []
    else:
        if phones not in pronouncing.phones_for_word(word):
            raise ValueError(phones + " not phones for" + word)
    if not phones:
        raise ValueError("phonemes string is empty")

    rp = pronouncing.rhyming_part(phones)
    search_combos = wildcard_mix_phones_regex_searches(rp, stress)
    rhymes = []
    for search in search_combos:
        rhymes += pronouncing.search(
            search + "( .{1,3}){0," + str(consonant_tail) + "}$")
    if rhymes:
        rhymes = unique(rhymes)
        if word in rhymes:
            rhymes.remove(word)
        return rhymes
    print("random general rhyme: tried all combos, didn't find anything!")
    return []

def random_general_rhyme(word, phones=None, search_option="end"):
    """ Return a list of rhymes where a random combination of phonemes match

    The conditions for a general rhyme between words are:
    (1) Any possible phonetic similarity between the final stressed vowel and
        subsequent phonemes.
    If phones argument not given, phones/pronunciation used will default to the
    first in the list of phones returned for word. If no rhyme is found, an
    empty list is returned.


    :param word: a word
    :param phones: specific CMUdict phonemes string for word (default None)
    :param search_option option for regex search. (default "end")
    :return: a list of rhymes for word, where specific rhyme is random
    """
    if phones is None:
        phones = first_phones_for_word(word)
        if phones == "":
            return []
    else:
        if phones not in pronouncing.phones_for_word(word):
            raise ValueError(phones + " not phones for +" + word)
    if not phones:
        raise ValueError("phonemes string is empty")
    rp = pronouncing.rhyming_part(phones)
    search_combos = wildcard_mix_phones_regex_searches(rp)
    while search_combos:
        search = random.choice(search_combos)
        if search_option == "end":
            rhymes = pronouncing.search(search + "$")
        elif search_option == "begin":
            rhymes = pronouncing.search("^" + search)
        elif search_option == "whole":
            rhymes = pronouncing.search("^" + search + "$")
        else:
            raise ValueError("search_option should be 'end', 'begin', or 'whole'")
        if rhymes:
            rhymes = unique(rhymes)
            if word in rhymes:
                rhymes.remove(word)
            return rhymes
        else:
            search_combos.remove(search)
    print("random general rhyme: tried all combos, didn't find anything!")
    return []


def random_match_phones(word, phones=None):
    """Returns words that match a random combination of phonemes

    This is like a random general rhyme, however instead of just the
    last syllable portion, it's the entire word.

    :param word: word that should be in the CMU Pronouncing Dictionary
    :param phones: specific phonemes to rhyme with (default None)
    :return: a word that shares a random combinations of phonemes
    """
    if phones is None:
        phones = first_phones_for_word(word)
        if phones == "":
            return []
    else:
        if phones not in pronouncing.phones_for_word(word):
            raise ValueError("phonemes and word don't match")
    if not phones:
        raise ValueError("phonemes string is empty")
    search_list = wildcard_mix_phones_regex_searches(phones)
    while search_list:
        search = random.choice(search_list)
        rhymes = pronouncing.search(search)
        if rhymes:
            rhymes = unique(rhymes)
            if word in rhymes:
                rhymes.remove(word)
            return rhymes
        else:
            search_list.remove(search)
    print("random general match phones: tried all combos, didn't find anything!")
    return []


def rhyme_same_stress(word):
    timeout_timer = 0
    # print('in the stress loop')
    while(True):
        phones = pronouncing.phones_for_word(word)
        phone = random.choice(phones)
        word_stress = pronouncing.stresses(phone)
        rhyme = rhyme_type_random(word)
        phones = pronouncing.phones_for_word(rhyme)
        for phone in phones:
            rhyme_stress = pronouncing.stresses(phone)
            if word_stress == rhyme_stress:
                return rhyme
        print(timeout_timer)
        if timeout_timer == 10:
            return rhyme
        timeout_timer += 1

def sim_word_for_phones(phones, word_list=[], sim_perc = 0.25):
	"""Finds a word that has shared phonemes to input phonemes

	:param phones: CMU Pronouncing Dictionary phonemes string
	:param word_list: list of words to limit search to
	:sim_perc: threshold for similarity between phones and words in words_list
	:return: word that has number of phonemes the same as phones
	"""
	search_combos = wildcard_mix_phones_regex_searches(phones)
	random.shuffle(search_combos)
	for sch in search_combos:
		sch_list = sch.split(" ")
		if sch_list.count(".{1,3}") < (1-sim_perc)*len(sch_list):
			matches = pronouncing.search("^" + sch + "$")
			if matches:
				matches = unique(matches)
				random.shuffle(matches)
				for m in matches:
						if word_list:
							if m in word_list:
								return(m)
						else:
							return(m)
	return None

def rhyme_type_random(word):
    rhyme_types = ['perfect', 'identical', 'random_match_phones', 'random_general']
    rhymes = []
    while rhyme_types:
        rt = random.choice(rhyme_types)
        if rt == 'perfect':
            rhymes = perfect_rhyme(word)
        elif rt == 'identical':
            rhymes = identical_rhyme(word)
        elif rt == 'random_match_phones':
            rhymes = random_match_phones(word)
        elif rt == 'random_general':
            rhymes = random_general_rhyme(word)
        if rhymes:
            a_rhyme = random.choice(rhymes)
            return a_rhyme
        else:
            rhyme_types.remove(rt)
    return []

def wildcard_mix_phones_regex_searches(phones, stress=False):
    """Generates all combinations of regex strings where phoneme in 'phones' is a wildcard ('.')

    e.g. ['HH IY1 R'],['HH IY1 .{1,3}'],['HH .{1,3} R'],
        ['.{1,3} IY1 R'], ...['.{1,3} .{1,3} .{1,3}']


    :param phones: CMU Pronouncing Dictionary phonemes string
    :param stress: if stress portion of vowel is included or nont (a '.')
    :return: list of regex search strings where phonemes replaced with wildcard
    """
    phones_list = phones.split()
    product_factors = []
    for phone in phones_list:
        flist = ['.{1,3}']
        if stress is False and check_if_vowel(phone):
            flist.append(phone[:2]+'.')  # ignore stress
        else:
            flist.append(phone)
        product_factors.append(flist)
    combos = list(itertools.product(*product_factors))
    combos.remove(combos[0])  # should be case where ['.', '.', ... '.']
    search_combos = [' '.join(list(item)) for item in combos]
    return search_combos

def get_last_word(text: str) -> str | None:
    """
    Extract the last word-like token from a string.
    Returns it in lowercase, or None if no word is found.
    """
    # Allow letters and apostrophes (you can tweak this if you want)
    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return None
    return tokens[-1].lower()


def build_rhyme_list_for_line(line: str) -> tuple[str | None, list[str]]:
    """
    Take a line of text, get its last word,
    then generate all rhyme types using rhyming functions:

    - rhyme
    - identical_rhyme
    - perfect_rhyme
    - near_rhyme

    Returns:
        (last_word, combined_rhyme_list)
    """
    word = get_last_word(line)
    if word is None:
        return None, []

    # Use rhyme functions
    rhymes_basic = rhyme(word)
    rhymes_identical = identical_rhyme(word)
    rhymes_perfect = perfect_rhyme(word)
    # rhymes_near = near_rhyme(word)

    # Combine and deduplicate
    all_rhymes = sorted(
        set(rhymes_basic)
        | set(rhymes_identical)
        | set(rhymes_perfect)
        # | set(rhymes_near)
    )

    return word, all_rhymes


def last_word_in_rhyme_list(source_line: str, other_line: str) -> bool:
    """
    Build the rhyme list from the source_line's last word,
    then check whether the last word of other_line is in that list.
    """
    _, rhyme_list = build_rhyme_list_for_line(source_line)
    target_word = get_last_word(other_line)

    if target_word is None:
        return False

    return target_word in rhyme_list

# Example usage
line1 = "The cat sat on the mat."
line2 = "I saw a bat."
line3 = "This line ends with dog."

base_word, rhyme_list = build_rhyme_list_for_line(line1)
print(f"Last word of line1: {base_word}")
print(f"Total rhymes found: {len(rhyme_list)}")
print("Some rhymes:", rhyme_list[:20])  # just a preview

print(f"\nDoes the last word of line2 rhyme with '{base_word}'?")
print(last_word_in_rhyme_list(line1, line2))  # likely True (bat)

print(f"\nDoes the last word of line3 rhyme with '{base_word}'?")
print(last_word_in_rhyme_list(line1, line3))  # likely False

def cheap_last_word(line: str) -> str | None:
  tokens = line.split(' ')
  if not tokens:
    return None
  return tokens[-1].lower()

def calculate_rhyming_percentage(original_stanza, generated_stanza, delimiter='\n'):
  original_lines = original_stanza.split(delimiter)
  generated_lines = generated_stanza.split(delimiter)

  # pad lines if needed with empty strings
  max_lines = max(len(original_lines), len(generated_lines))
  original_lines += [''] * (max_lines - len(original_lines))
  generated_lines += [''] * (max_lines - len(generated_lines))

  rhyme_bools_by_line = []

  for (o,g) in zip(original_lines, generated_lines):
    if o == '' and g == '':
      rhyme_bools_by_line.append(True)
    # xor
    elif (o == '') ^ (g == ''):
      rhyme_bools_by_line.append(False)
    else:
      last_word = cheap_last_word(o)
      if last_word in cached_rhyme_structures:
        rhyme_list = cached_rhyme_structures[last_word]
        # print(rhyme_list)
      else:
        _, rhyme_list = build_rhyme_list_for_line(o)
        cached_rhyme_structures[last_word] = rhyme_list
        for r in rhyme_list:
          cached_rhyme_structures[r] = rhyme_list
      rhyme_bools_by_line.append(last_word_in_rhyme_list(o, g))

  return sum(rhyme_bools_by_line) / len(rhyme_bools_by_line)

import time

global cached_rhyme_structures
cached_rhyme_structures = {} # important that this is only initialized once

start = time.time()
print(calculate_rhyming_percentage('hello test test', 'hello best best'))
lap = time.time()
print("Time 1: " + str(lap-start))
print(calculate_rhyming_percentage('Hey I just met you \n and this is crazy \n but here\'s my number \n so call me maybe',"So here’s my number,\n it’s on this baby—\nI think this joke’s \n gone too far, maybe?"))
lap2 = time.time()
print("Time 2: " + str(lap2-lap))
print(calculate_rhyming_percentage('Hey I just met you \n and this is crazy \n but here\'s my number \n so call me maybe',"So here’s my number,\n it’s on this baby—\nI think this joke’s \n gone too far, maybe?"))
lap3 = time.time()
print("Time 3: " + str(lap3-lap2))
print(calculate_rhyming_percentage('Hey I just met ado \n and this is blasi \n but here\'s my slumber \n so call me baby',"So here’s my number,\n it’s on this baby—\nI think this joke’s \n gone too far, maybe?"))
print(calculate_rhyming_percentage('Hey I just met ado \n and this is blasi \n but here\'s my slumber \n so call me baby',"So here’s my number,\n it’s on this baby—\nI think this joke’s \n gone too far, maybe?"))
print(calculate_rhyming_percentage('Hey I just met ado \n and this is blasi \n but here\'s my slumber \n so call me baby',"So here’s my number,\n it’s on this baby—\nI think this joke’s \n gone too far, maybe?"))
lap4 = time.time()
print("Time 4: " + str(lap4-lap3))
print("Overall time:")
print(lap4-start)

