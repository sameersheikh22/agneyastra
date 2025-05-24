import spacy
from itertools import combinations
from collections import Counter

class KeywordGeneratorAgent:
    """
    A class to generate keywords from extracted arguments using spaCy.
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
            print(f"Falling back to a blank English model ('en'). Processing quality will be affected.")
            self.nlp = spacy.blank("en")
            # Add sentence boundary detection and a tagger if using a blank model
            if not self.nlp.has_pipe("sentencizer"):
                 self.nlp.add_pipe("sentencizer")
            # Basic POS tagging might not be available or accurate in a blank model without further components.
            # For keyword extraction based on POS, a full model is highly recommended.
            # We'll proceed, but results might be poor with a blank model.

    def generate_keywords(self, extracted_arguments: dict, top_n_terms: int = 7, max_keywords: int = 20) -> list:
        """
        Generates keywords from the research angle summary.

        Args:
            extracted_arguments: A dictionary containing summaries, expected to have
                                 a 'research_angle_summary' key.
            top_n_terms: The number of most frequent single terms to use for combinations.
            max_keywords: The maximum number of keywords to return.

        Returns:
            A list of generated keyword strings.
        """
        research_angle_summary = extracted_arguments.get("research_angle_summary")

        if not research_angle_summary or not isinstance(research_angle_summary, str) or not research_angle_summary.strip():
            return ["Missing or invalid research angle summary"]

        if not self.nlp:
            return ["spaCy model not available for keyword generation"]

        doc = self.nlp(research_angle_summary)
        key_terms_lemmas = []
        
        # Extract nouns, proper nouns, adjectives, and main verbs
        # Using lemmas for normalization
        for token in doc:
            if not token.is_stop and not token.is_punct:
                if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]:
                    key_terms_lemmas.append(token.lemma_.lower())

        if not key_terms_lemmas:
            return ["No key terms extracted from summary"]

        # Count frequencies of individual lemma terms
        term_counts = Counter(key_terms_lemmas)
        
        # Select top N most frequent terms for generating combinations
        # If there are fewer unique terms than top_n_terms, use all of them.
        num_terms_for_combination = min(top_n_terms, len(term_counts))
        most_common_single_terms = [term for term, count in term_counts.most_common(num_terms_for_combination)]
        
        # Start with the most common single terms
        generated_keywords = list(most_common_single_terms)

        # Generate 2-word combinations from the most_common_single_terms
        if len(most_common_single_terms) >= 2:
            for combo in combinations(most_common_single_terms, 2):
                generated_keywords.append(" ".join(sorted(combo))) # Sort for consistent order

        # Generate 3-word combinations from the most_common_single_terms
        if len(most_common_single_terms) >= 3:
            for combo in combinations(most_common_single_terms, 3):
                generated_keywords.append(" ".join(sorted(combo))) # Sort for consistent order
        
        # Remove duplicates while preserving order of first appearance (for single terms)
        # and then add combinations which might create new duplicates if not handled.
        # Using dict.fromkeys for unique keywords, then converting back to list.
        # This also helps in case combinations recreate single keywords already present.
        unique_keywords = list(dict.fromkeys(generated_keywords))
        
        return unique_keywords[:max_keywords]
