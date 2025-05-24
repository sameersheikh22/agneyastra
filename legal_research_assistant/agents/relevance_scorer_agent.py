import spacy

class RelevanceScorerAgent:
    """
    A class to score the relevance of found sources against the research angle
    using spaCy similarity and Tavily scores.
    """
    def __init__(self, model_name="en_core_web_md", similarity_weight=0.6, tavily_score_weight=0.4):
        """
        Initializes the agent and loads the spaCy model.

        Args:
            model_name (str): The name of the spaCy model with word vectors to load.
            similarity_weight (float): Weight for spaCy similarity score in final calculation.
            tavily_score_weight (float): Weight for Tavily's search score in final calculation.
        """
        self.similarity_weight = similarity_weight
        self.tavily_score_weight = tavily_score_weight
        self.model_loaded_successfully_with_vectors = False
        
        try:
            self.nlp = spacy.load(model_name)
            if self.nlp.has_pipe("tok2vec") or self.nlp.has_pipe("transformer"): # Check if it has vectors
                self.model_loaded_successfully_with_vectors = True
                print(f"Successfully loaded spaCy model '{model_name}' with word vectors.")
            else:
                # TODO: Replace print with logging
                print(f"Warning: spaCy model '{model_name}' loaded, but it does not seem to have word vectors. "
                      f"Similarity scores will be based on tagger/parser features or be 0.")
                # Attempt to load a fallback model WITH vectors if the primary choice didn't have them but loaded.
                if model_name != "en_core_web_sm": # Avoid infinite loop if sm also fails
                    try:
                        print(f"Attempting to fall back to 'en_core_web_sm' for basic processing...")
                        self.nlp = spacy.load("en_core_web_sm")
                        print("Warning: Fallback to 'en_core_web_sm' successful, but this model lacks word vectors. "
                              "Similarity scores will be 0 or less meaningful.")
                    except OSError:
                        print("Error: Fallback model 'en_core_web_sm' also not found. "
                              "Please download a spaCy model (e.g., 'en_core_web_md') to enable similarity scoring.")
                        self.nlp = spacy.blank("en") # Last resort
                        if not self.nlp.has_pipe("sentencizer"): self.nlp.add_pipe("sentencizer")

        except OSError:
            # TODO: Replace print with logging
            print(f"spaCy model '{model_name}' not found. Please download it by running:")
            print(f"python -m spacy download {model_name}")
            print("Attempting to fall back to 'en_core_web_sm'...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
                # TODO: Replace print with logging
                print("Warning: Fallback to 'en_core_web_sm' successful, but this model lacks word vectors. "
                      "Similarity scores will be 0 or less meaningful.")
            except OSError:
                # TODO: Replace print with logging
                print("Error: Fallback model 'en_core_web_sm' also not found. "
                      "Similarity scoring will be disabled. Please download a spaCy model.")
                self.nlp = spacy.blank("en") # Last resort, provides tokenization and sentence boundaries
                if not self.nlp.has_pipe("sentencizer"): self.nlp.add_pipe("sentencizer")


    def score_sources(self, sources: list[dict], research_angle_summary: str) -> list[dict]:
        """
        Scores the relevance of a list of sources based on a research angle summary.

        Args:
            sources: A list of source dictionaries (from SourceCrawlerAgent/CitationChainerAgent).
            research_angle_summary: A summary of the research angle.

        Returns:
            A list of sources, sorted by 'final_relevance_score' and augmented with
            'similarity_to_research_angle' and 'final_relevance_score'.
        """
        if not self.nlp:
            # TODO: Replace print with logging
            print("Error: spaCy NLP model not available in RelevanceScorerAgent. Cannot score sources.")
            for source in sources:
                source['similarity_to_research_angle'] = 0.0
                source['final_relevance_score'] = source.get('score', 0.0) # Use Tavily score if available
            # Sort by Tavily score if NLP model failed
            return sorted(sources, key=lambda s: s.get('final_relevance_score', 0.0), reverse=True)

        if not research_angle_summary or not research_angle_summary.strip():
            # TODO: Replace print with logging
            print("Warning: Research angle summary is empty. Similarity scores will be 0.")
            research_angle_doc = self.nlp("") # Empty doc
        else:
            research_angle_doc = self.nlp(research_angle_summary)
            if not research_angle_doc.has_vector and research_angle_doc.vector_norm == 0:
                 # TODO: Replace print with logging
                print("Warning: Research angle summary doc has no vector. Similarity scores may be 0.")


        for source in sources:
            abstract = source.get('abstract', '')
            similarity = 0.0

            if not abstract or not abstract.strip() or len(abstract.split()) < 10: # Min 10 words for meaningful similarity
                # TODO: Replace print with logging (for very short abstracts)
                # print(f"Notice: Abstract for '{source.get('title', 'N/A')}' is too short or missing. Assigning low similarity.")
                similarity = 0.0 # Low similarity for short/missing abstracts
            elif not self.model_loaded_successfully_with_vectors:
                similarity = 0.0 # Model without vectors, similarity is not reliable
            else:
                try:
                    abstract_doc = self.nlp(abstract)
                    if abstract_doc.has_vector and abstract_doc.vector_norm != 0 and \
                       research_angle_doc.has_vector and research_angle_doc.vector_norm != 0:
                        similarity = research_angle_doc.similarity(abstract_doc)
                        # Ensure similarity is float, not numpy.float32/64 for broader compatibility (e.g. JSON)
                        similarity = float(similarity) 
                    else:
                        # Handle cases where one or both docs have no vector (e.g. all OOV words, or empty)
                        similarity = 0.0
                except Exception as e:
                    # TODO: Replace print with logging
                    print(f"Error calculating similarity for '{source.get('title', 'N/A')}': {e}")
                    similarity = 0.0
            
            source['similarity_to_research_angle'] = round(similarity, 4)

            # Retrieve Tavily score (already normalized 0-1 by Tavily)
            tavily_search_score = source.get('score', 0.0) 

            # Combine scores
            # If model with vectors wasn't loaded, rely more on Tavily score or just use it.
            current_similarity_weight = self.similarity_weight if self.model_loaded_successfully_with_vectors else 0.1
            current_tavily_weight = self.tavily_score_weight if self.model_loaded_successfully_with_vectors else 0.9
            
            final_score = (current_similarity_weight * similarity) + \
                          (current_tavily_weight * tavily_search_score)
            source['final_relevance_score'] = round(final_score, 4)

        # Sort sources by the final relevance score in descending order
        sorted_sources = sorted(sources, key=lambda s: s.get('final_relevance_score', 0.0), reverse=True)
        
        return sorted_sources
