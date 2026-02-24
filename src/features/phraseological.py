import math
from collections import defaultdict

import numpy as np
import pandas as pd


class MiCalculator:
    """Mutual Information calculator for dependency relations.

    MI measures how predictable a dependent word is given its head word.
    MI = log2(P(head,dep) / (P(head) * P(dep)))
    """

    DEFAULT_RELATIONS = ("amod", "dobj", "advmod")

    def __init__(self, reference_grams: pd.DataFrame, relations=None):
        self.relations = set(relations or self.DEFAULT_RELATIONS)
        self.dep_counts = dict(
            zip(
                zip(
                    reference_grams["head_lemma"],
                    reference_grams["dependent_lemma"],
                    reference_grams["relation"],
                ),
                reference_grams["count"],
            )
        )
        self.head_marginals = (
            reference_grams.groupby("head_lemma")["count"].sum().to_dict()
        )
        self.dep_marginals = (
            reference_grams.groupby("dependent_lemma")["count"].sum().to_dict()
        )
        self.total_deps = reference_grams["count"].sum()

    def __call__(self, doc) -> dict:
        rel_mis = defaultdict(list)
        for token in doc:
            if token.dep_ not in self.relations:
                continue
            head_lemma = token.head.lemma_.lower()
            dep_lemma = token.lemma_.lower()
            relation = token.dep_
            pair = (head_lemma, dep_lemma, relation)

            joint_count = self.dep_counts.get(pair, 0)
            if joint_count == 0:
                continue

            p_xy = joint_count / self.total_deps
            p_x = self.head_marginals.get(head_lemma, 0) / self.total_deps
            p_y = self.dep_marginals.get(dep_lemma, 0) / self.total_deps

            mi = math.log2(p_xy / (p_x * p_y))
            rel_mis[relation].append(mi)

        return {
            f"{rel}_mi": (np.mean(rel_mis[rel]) if rel in rel_mis else np.nan)
            for rel in self.relations
        }
