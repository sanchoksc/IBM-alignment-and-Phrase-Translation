from nltk.tokenize import word_tokenize
from nltk.translate.ibm1 import IBMModel1
from nltk.translate.ibm2 import IBMModel2
from nltk.translate import AlignedSent

def nltk_ibm_one(data, iter=5):
    dual_text = []
    for d_i in range(len(data)):
        fr_sent = word_tokenize(data[d_i]['fr'])
        eng_sent = word_tokenize(data[d_i]['en'])
        dual_text.append(AlignedSent(fr_sent, eng_sent))
    ibm_one = IBMModel1(dual_text, iter)
    print("Probability score for the: ")
    print(ibm_one.translation_table['maison']['house'])

def nltk_ibm_two(data, iter=5):
    dual_text = []
    for d_i in range(len(data)):
        fr_sent = word_tokenize(data[d_i]['fr'])
        eng_sent = word_tokenize(data[d_i]['en'])
        dual_text.append(AlignedSent(fr_sent, eng_sent))
    ibm_two = IBMModel2(dual_text, iter)
    print("Probability score for the: ")
    print(ibm_two.translation_table['maison']['house'])
