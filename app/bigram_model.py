from collections import defaultdict, Counter
import random
import re


class BigramModel:
    """
    A simple Bigram language model.

    The model learns the probability of the next word
    based on the current word.
    """

    def __init__(self, corpus, frequency_threshold=None):
        """
        Initialize the Bigram model using a corpus.

        Args:
            corpus: A list of text strings.
            frequency_threshold: Minimum frequency required
                                 for a word to be included.
        """

        # Combine all sentences in the corpus into one text
        self.text = " ".join(corpus)

        # Analyze the corpus and build bigram probabilities
        self.vocab, self.bigram_probs = self.analyze_bigrams(
            self.text,
            frequency_threshold
        )


    def simple_tokenizer(self, text, frequency_threshold=None):
        """
        Convert text to lowercase and split it into words.
        """

        # Convert to lowercase and extract words using regex
        tokens = re.findall(r"\b\w+\b", text.lower())

        # If no frequency threshold is specified,
        # keep all tokens
        if not frequency_threshold:
            return tokens

        # Count word frequencies
        word_counts = Counter(tokens)

        # Keep words that meet the frequency threshold
        filtered_tokens = [
            token
            for token in tokens
            if word_counts[token] >= frequency_threshold
        ]

        return filtered_tokens


    def analyze_bigrams(self, text, frequency_threshold=None):
        """
        Analyze text and compute bigram probabilities.
        """

        # Tokenize the text
        words = self.simple_tokenizer(
            text,
            frequency_threshold
        )

        # Create bigrams
        bigrams = list(zip(words[:-1], words[1:]))

        # Count bigram frequencies
        bigram_counts = Counter(bigrams)

        # Count unigram frequencies
        unigram_counts = Counter(words)

        # Store bigram probabilities
        bigram_probs = defaultdict(dict)

        # Calculate P(word2 | word1)
        for (word1, word2), count in bigram_counts.items():

            bigram_probs[word1][word2] = (
                count / unigram_counts[word1]
            )

        return list(unigram_counts.keys()), bigram_probs


    def generate_text(self, start_word, num_words=20):
        """
        Generate text using the learned bigram probabilities.

        Args:
            start_word: The first word of the generated text.
            num_words: Maximum number of words to generate.

        Returns:
            Generated text as a string.
        """

        # Convert starting word to lowercase
        current_word = start_word.lower()

        # Start generated text with the input word
        generated_words = [current_word]

        # Generate the remaining words
        for _ in range(num_words - 1):

            # Find possible words that can follow current_word
            next_words = self.bigram_probs.get(current_word)

            # Stop if the current word has no known next word
            if not next_words:
                break

            # Randomly choose the next word
            # according to the bigram probabilities
            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values())
            )[0]

            # Add the word to the generated text
            generated_words.append(next_word)

            # Move to the next word
            current_word = next_word

        return " ".join(generated_words)