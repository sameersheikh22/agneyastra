import spacy
from heapq import nlargest

class ArgumentExtractionAgent:
    """
    A class to extract arguments from a base paper based on a research angle using spaCy.
    """
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initializes the agent and loads the spaCy model.

        Args:
            model_name (str): The name of the spaCy model to load.
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model '{model_name}' not found. Please download it by running:")
            print(f"python -m spacy download {model_name}")
            # Fallback to a blank English model if the specified model isn't available
            # This allows the program to run but with potentially reduced functionality.
            print(f"Falling back to a blank English model ('en'). Processing quality will be affected.")
            self.nlp = spacy.blank("en")
            # Add sentence boundary detection for the blank model
            if not self.nlp.has_pipe("sentencizer"):
                 self.nlp.add_pipe("sentencizer")


    def _extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Performs extractive summarization on the given text.

        Args:
            text: The input string to summarize.
            num_sentences: The number of top sentences to include in the summary.

        Returns:
            A string containing the summary.
        """
        if not self.nlp or not text.strip():
            return "Text processing model not available or empty input."

        doc = self.nlp(text)
        
        # Handle cases where self.nlp might be a blank model without proper stop words
        stopwords = list(self.nlp.Defaults.stop_words) if hasattr(self.nlp.Defaults, "stop_words") else []

        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in stopwords and word.is_alpha:
                if word.text.lower() not in word_frequencies:
                    word_frequencies[word.text.lower()] = 1
                else:
                    word_frequencies[word.text.lower()] += 1

        if not word_frequencies: # Handle text with no processable words (e.g. only stopwords or symbols)
            return "Could not generate summary due to lack of processable words."

        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency

        sentence_tokens = [sent for sent in doc.sents]
        if not sentence_tokens:
            return "Could not generate summary as no sentences were detected."

        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        
        if not sentence_scores: # All sentences might have no scorable words
             return "Could not score sentences for summary."


        select_length = min(num_sentences, len(sentence_tokens))
        summary_sentences = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        summary = " ".join([sent.text.strip() for sent in summary_sentences])
        return summary

    def extract_arguments(self, base_paper_text: str, research_angle_text: str) -> dict:
        """
        Extracts arguments (summaries) from the base paper relevant to the research angle.

        Args:
            base_paper_text: The text of the base paper.
            research_angle_text: The text describing the research angle.

        Returns:
            A dictionary containing the extracted summaries.
        """
        base_paper_summary = self._extractive_summary(base_paper_text, num_sentences=5)
        research_angle_summary = self._extractive_summary(research_angle_text, num_sentences=3)

        return {
            "base_paper_summary": base_paper_summary,
            "research_angle_summary": research_angle_summary
        }
