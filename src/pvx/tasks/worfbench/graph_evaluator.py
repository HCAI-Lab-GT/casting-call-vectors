"""
Pulled from WorFBench GitHub

WorfBench/evaluator/graph_evaluator.py
"""

from collections import defaultdict
import json
from numpy import mean

# import evaluate
import itertools
import networkx as nx
import numpy as np
import copy
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
from typing import List, Dict


def all_topological_sorts(graph: Dict[str, List[str]]) -> List[List[str]]:
    # Return the nodes if there are 10 or more nodes
    if len(graph["nodes"]) >= 10:
        return [graph["nodes"]]

    # Create a directed graph
    G = nx.DiGraph()

    # Map to hold original node names by their indices
    original_nodes = graph["nodes"]

    # Add nodes using their indices (0, 1, 2, ..., n)
    G.add_nodes_from(range(len(original_nodes)))

    # Add edges using the indices
    edges_with_indices = [(u, v) for u, v in graph["edges"]]
    G.add_edges_from(edges_with_indices)

    # Get all possible topological sorts
    all_sorts = list(nx.all_topological_sorts(G))

    # Convert sorted indices back to original node names and remove "START" and "END"
    filtered_sorts = []
    for sort in all_sorts:
        filtered_sort = [
            original_nodes[i] for i in sort if original_nodes[i] not in ["START", "END"]
        ]
        filtered_sorts.append(filtered_sort)

    return filtered_sorts[:20]  # Return the first 20 sorted lists


def largest_connected_component(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    connected_components = nx.connected_components(G)
    largest_component = max(connected_components, key=len)

    return len(largest_component)


def match_node(
    pred_nodes: List[str], gt_nodes: List[str], sentence_model: object, match_threshold=0.6
) -> dict:
    len_pred = len(pred_nodes)
    len_gt = len(gt_nodes)
    bert_score_matrix = np.zeros((len_pred, len_gt))
    # bert_model = eval_model
    # sentence_model = SentenceTransformer(bert_model)

    node_pred_emb = sentence_model.encode(pred_nodes, convert_to_tensor=True)
    node_gt_emb = sentence_model.encode(gt_nodes, convert_to_tensor=True)

    node_cosine_scores = np.maximum(util.cos_sim(node_pred_emb, node_gt_emb).cpu().numpy(), 0)

    for i in range(len_pred):
        for j in range(len_gt):
            bert_score_matrix[i][j] = node_cosine_scores[i][j]
    G = nx.Graph()

    for i in range(len_pred):
        for j in range(len_gt):
            if bert_score_matrix[i][j] > match_threshold:
                G.add_edge(i, str(j), weight=bert_score_matrix[i][j])
    max_weight_matching = nx.max_weight_matching(G)
    pred_to_gt_mapping = dict()
    for key in max_weight_matching:
        if type(key[0]) == int:
            pred_to_gt_mapping[int(key[0])] = int(key[1])
        else:
            pred_to_gt_mapping[int(key[1])] = int(key[0])

    # If a prediction node does not match any golden answer node, we mark the node as -1.
    for i in range(len_pred):
        if i not in pred_to_gt_mapping:
            pred_to_gt_mapping[i] = -1
    return pred_to_gt_mapping


def t_eval_graph(
    pred_graph: Dict[str, List[str]], gt_graph: Dict[str, List[str]], sentence_model: object
) -> Dict[str, float]:
    pred_nodes = pred_graph["nodes"]
    gt_nodes = gt_graph["nodes"]
    pred_to_gt_mapping = match_node(pred_nodes, gt_nodes, sentence_model)
    if len(pred_nodes) == 0 or len(gt_nodes) == 0:
        return {"precision": 0, "recall": 0, "f1_score": 0}

    # print(f"pred_to_gt_mapping:{pred_to_gt_mapping}")

    # find mathced nodes and edges
    pred_edges = pred_graph["edges"]
    gt_edges = gt_graph["edges"]
    matched_pred_nodes = []
    matched_gt_nodes = []
    for k, v in pred_to_gt_mapping.items():
        if v != -1:
            matched_pred_nodes.append(k)
            matched_gt_nodes.append(v)
    matched_pred_edges = []
    for pred_edge in pred_edges:
        if pred_edge[0] in matched_pred_nodes and pred_edge[1] in matched_pred_nodes:
            # connect the matched nodes
            matched_pred_edges.append((pred_edge[0], pred_edge[1]))
    matched_gt_edges = []
    for gt_edge in gt_edges:
        if gt_edge[0] in matched_gt_nodes and gt_edge[1] in matched_gt_nodes:
            # connect the matched nodes
            matched_gt_edges.append((gt_edge[0], gt_edge[1]))
    public_edges = []
    for mpe in matched_pred_edges:
        if mpe in matched_gt_edges:
            public_edges.append(mpe)
    # calculate the number of connected components
    pred_connected_components = largest_connected_component(matched_pred_nodes, public_edges)
    gt_connected_components = largest_connected_component(matched_gt_nodes, public_edges)

    precision = pred_connected_components / len(pred_nodes)
    recall = gt_connected_components / len(gt_nodes)
    f1_score = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1_score": f1_score}


def t_eval_nodes(
    pred_graph: Dict[str, List[str]], gt_graph: Dict[str, List[str]], sentence_model: object
) -> Dict[str, float]:
    try:
        all_gold_trajects = all_topological_sorts(gt_graph)
    except Exception as e:
        print(e, gt_graph)

    pred_nodes = [
        pred_node for pred_node in pred_graph["nodes"] if pred_node not in ["START", "END"]
    ]
    max_f1 = {"precision": 0, "recall": 0, "f1_score": 0}
    for gold_nodes in all_gold_trajects:
        result = t_eval_plan(pred_nodes, gold_nodes, sentence_model)
        if result["f1_score"] > max_f1["f1_score"]:
            max_f1 = result
    return max_f1


def t_eval_plan(
    pred_plan: List[str], gt_plan: List[str], eval_model: object, order: bool = True
) -> dict:
    """
    Calculate the similarity between predicted plan and golden answer,
    A plan can be regarded a sequence of actions, and each action has a name and args.
    Firstly, use bertscore to calculate pointwise similarity by:
        similarity(u, v) = bertscore(u.name, v.name) * name_weight + bertscore(u.args, v.args) * args_weight;
    Secondly, use Hungarian matching to match the points;
    Finally, use LIS to calculate the number of matched nodes.
    """
    pred_to_gt_mapping = match_node(pred_plan, gt_plan, eval_model)

    # print(f"pred_to_gt_mapping:{pred_to_gt_mapping}")
    len_pred = len(pred_plan)
    len_gt = len(gt_plan)
    if order:
        dp = np.ones(len_pred)
        for i in range(len_pred):
            for j in range(i):
                if pred_to_gt_mapping[i] == -1 or pred_to_gt_mapping[j] == -1:
                    continue
                if pred_to_gt_mapping[i] > pred_to_gt_mapping[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        correct_count = int(max(dp))

        recall, precision = correct_count / len(gt_plan), correct_count / len(pred_plan)
        f1_score = 2 * recall * precision / (recall + precision)
        result = {"precision": precision, "recall": recall, "f1_score": f1_score}
    else:
        correct_count = 0
        for i in range(len_pred):
            if pred_to_gt_mapping[i] != -1:
                correct_count += 1
        fail_recall = 0
        for i in range(len_gt):
            if i not in pred_to_gt_mapping.values():
                fail_recall += 1
        recall = (len_gt - fail_recall) / len_gt
        precision = correct_count / len_pred
        if correct_count == 0:
            f1_score = 0
        else:
            f1_score = 2 * recall * precision / (recall + precision)
        result = {"precision": precision, "recall": recall, "f1_score": f1_score}
    return result


if __name__ == "__main__":
    pred_graph = {
        "nodes": [
            "START",
            "Find the statutory formula Calculate the K value.",
            "Compute each parent's monthly net disposable income.",
            "END",
        ],
        "edges": [(0, 1), (1, 2), (2, 3)],
    }
    gt_graph = {
        "nodes": [
            "START",
            "go to toilet",
            "take dishsponge from toilet",
            "go to sinkbasin",
            "clean dishsponge with sinkbasin",
            "go to toilet",
            "put dishsponge in/on toilet.",
            "END",
        ],
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
    }
    eval_model = "all-mpnet-base-v2"
    eval_model = SentenceTransformer(eval_model)
    # print(t_eval_plan(pred_graph["nodes"],gt_graph["nodes"],eval_model))
    print(t_eval_nodes(pred_graph, gt_graph, eval_model))
