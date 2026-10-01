import spacy

# Load the large English model
nlp = spacy.load("en_core_web_lg")


def calculate_embedding(input_word):
    """Calculate the word embedding for an input word."""
    word = nlp(input_word)
    return word.vector.tolist()