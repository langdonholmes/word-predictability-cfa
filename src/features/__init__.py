from .lexical import lexical_density, log_mean_token_freq, mean_word_length, mtld
from .syntactic import clauses_per_tunit, mean_dep_distance, mod_per_nom, words_per_sentence
from .phraseological import MiCalculator
from .cohesion import connective_density, content_word_overlap, sentence_similarity


def calculate_all_features(doc, token_freq, total_tokens, mi_calculator) -> dict:
    """Calculate all 13 linguistic features for a single spaCy Doc."""
    features = {}

    # Lexical (4)
    features["mtld"] = mtld(doc)
    features["lexical_density"] = lexical_density(doc)
    features["log_mean_token_freq"] = log_mean_token_freq(doc, token_freq, total_tokens)
    features["mean_word_length"] = mean_word_length(doc)

    # Syntactic (4)
    features["words_per_sentence"] = words_per_sentence(doc)
    features["clauses_per_tunit"] = clauses_per_tunit(doc)
    features["mod_per_nom"] = mod_per_nom(doc)
    features["mean_dep_distance"] = mean_dep_distance(doc)

    # Phraseological (3) -- MI scores
    mi_scores = mi_calculator(doc)
    features.update(mi_scores)

    # Cohesion (3)
    features["content_word_overlap"] = content_word_overlap(doc)
    features["connective_density"] = connective_density(doc)
    features["sentence_similarity"] = sentence_similarity(doc)

    return features
