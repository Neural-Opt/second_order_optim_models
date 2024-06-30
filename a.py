
from sacrebleu import corpus_bleu
import numpy as np
reference = [['Wiederaufnahme der Sitzungsperiode']]
hypothesis = ['Wiederaufnahme der Sitzungsperiode']

bleu_score = corpus_bleu(hypothesis, reference,smooth_method='exp')
print(f"BLEU score: {np.mean(np.array(bleu_score.precisions))}")