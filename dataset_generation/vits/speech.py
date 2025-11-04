#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shell
import re
#~ from phonemizer import phonemize
#~ from phonemizer.separator import Separator

voiced_consonant = ['m','n','J','b','d','g','jj','I','L','l','r','rr','B','D','G','gj']
vowel = ['a','e','i','j','o','u','w']
unvoiced = ['p', 't', 'c', 'k', 'tS', 'ts', 'tz', 'f', 'T', 's', 'z', 'S', 'x']
all_phonemes = 	voiced_consonant + vowel + unvoiced

def modulo1y2(src, mode='Spell', PhTSimple='y', language='eu', keep_chars = None, verbose = False):
	if src != '' and src != '.':
		if language in ['eu','es']:
			command = 'echo \''+ src + '\' | iconv -f UTF-8 -t ISO-8859-1 | ' + './modulo1y2' + ' '
			#command = 'echo \''+ src + '\' | ' + 'modulo1y2.exe' + ' '
			command += '-HDic=' + os.path.join('dict', language + '_dic') + ' '
			command += '-Lang=' + language + ' '
			command += '-TxtMode=' + mode + ' '
			command += '-PhTSimple=' + PhTSimple + ' '
			command +=  '| iconv -f ISO-8859-1 -t UTF-8'
			if(verbose):
				print(command)
			# ACORDAR QUITAR
			proc_output = os.system(command)
			output = proc_output.output()
			#output= ["k a j - S 'o"]
	else:
		output = []
	return output


def alphanumeric_clean(string,add='',remove=''):
	if '_' in add:
		remove_reg = remove
	else:
		remove_reg = '_' + remove
	
	if remove_reg != '':
		remove_reg = '|['+remove_reg+']'
		
	string_replaced = re.sub(r'([^\s\w'+add+']'+remove_reg+')','',string)
	return string_replaced


def modulo1(src_str, mode='Spell', PhTSimple='y', language='eu', silences = True, verbose = False):
	src = src_str
	if src != '' and src != '.':
		src = re.sub(r'\([^)]*\)', '', src)
		if language in ['eu','es']:
			src_string = alphanumeric_clean(src,add='!(),.:;?¿¡')
			if '\'' in src_string:
				src_string = src_string.replace('\'','\'\\\'\'')
			command = 'echo \''+ src_string + '\' | iconv -f UTF-8 -t ISO-8859-1 | ' + './modulo1' + ' '
			command += '-HDic=' + os.path.join('dict', language + '_dic')+' '
			command += '-Lang=' + language + ' '
			command +=  '| iconv -f ISO-8859-1 -t UTF-8'
			if(verbose):
				print(command)
			proc_output = shell.shell(command, capture_output=True, stderr_enc='iso-8859-1')
			output = proc_output.output
			phrases = []
			phrase = []
			for i, l in enumerate(output):
				if l == '':
					phrases.append(phrase)
					phrase = []
				else:
					phrase.append(l)
				if i == len(output) - 1:
					phrases.append(phrase)
			words = []
			for phrase in phrases:
				for i, line in enumerate(phrase):
					pho_split = line.split('/')
					if silences and i != 0:
						if(pho_split[7] == 'p'):
							words.append('_')
					words.append(pho_split[2].rstrip(' '))
			output = words
	else:
		output = []
	return output
