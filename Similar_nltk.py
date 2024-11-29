similar_consonants_nltk = {
    # 辅音相似表
    'P': ['B', 'M'], 'B': ['P', 'M'], 'M': ['B', 'P'],
    'F': ['V'], 'V': ['F'],
    'TH': ['DH'], 'DH': ['TH'],
    'T': ['D', 'S'], 'D': ['T', 'Z'], 'S': ['Z', 'SH'], 'Z': ['S', 'ZH'], 'N': ['D', 'T'],
    'SH': ['ZH', 'S'], 'ZH': ['SH', 'Z'],
    'K': ['G', 'NG'], 'G': ['K', 'NG'], 'NG': ['K', 'G'],
    'HH': ['']
}

similar_vowels_nltk = {
    # 元音相似表
    'IY1': ['IH1', 'EY1', 'EH1'], 'IH1': ['IY1', 'EH1', 'EY1'],
    'EY1': ['EH1', 'IY1', 'AE1'], 'EH1': ['AE1', 'IH1', 'EY1'],
    'AE1': ['EH1', 'AY1', 'AA1'], 'AA1': ['AE1', 'AO1', 'AH1'],
    'AO1': ['AA1', 'AW1', 'OW1'], 'AH1': ['AA1', 'EH1', 'AO1'],
    'UH1': ['UW1', 'AH1', 'OW1'], 'UW1': ['UH1', 'OW1', 'AO1'],
    'OW1': ['AO1', 'UH1', 'UW1']
}