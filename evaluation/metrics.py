from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import textstat

def bleu_score(references, hypothesis):
    bleu = sentence_bleu(references, hypothesis)
    return bleu

def rouge_score(references, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, references)
    rouge_1 = scores[0]["rouge-1"]["f"]
    rouge_2 = scores[0]["rouge-2"]["f"]
    rouge_l = scores[0]["rouge-l"]["f"]
    return rouge_1, rouge_2, rouge_l

def precision_recall(relevant_sentences, retrieved_sentences):
    true_positives = len(set(relevant_sentences).intersection(retrieved_sentences))
    precision = true_positives / len(retrieved_sentences)
    recall = true_positives / len(relevant_sentences)
    return precision, recall



def readability(predicted_text):
  """
  Evaluates the ease of understanding using the Flesch-Kincaid reading ease score.

  Args:
    predicted_text: The generated text.

  Returns:
    Flesch-Kincaid reading ease score (float).
  """
  return textstat.flesch_reading_ease(predicted_text)