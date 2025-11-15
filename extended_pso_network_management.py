import numpy as np
import networkx as nx
from pyswarms.multi import GlobalBestPSO
import requests
import json
import time

# -------------------
# Network Generation
# -------------------
NUM_NODES = 100
G = nx.erdos_renyi_graph(NUM_NODES, 0.05)  # Sparse large net

# Mock: generate scan results mapping nodes to CVEs (real use: parse NESSUS or OpenVAS)
nodes_with_cves = np.random.choice(G.nodes, size=int(NUM_NODES*0.2), replace=False)
node_cve_map = {node: [f'CVE-2022-{np.random.randint(100,999)}'] for node in nodes_with_cves}

# -------------------
# Real CVE Data Fetch (NVD API)
# -------------------
def fetch_cve_details(cve_id):
    url = f"https://services.nvd.nist.gov/rest/json/cve/1.0/{cve_id}"
    try:
        resp = requests.get(url)
        data = resp.json()
        # CVSS score and exploitability
        impact = data.get("result", {}).get("CVE_Items", [{}])[0].get("impact", {})
        cvss = impact.get("baseMetricV3", {}).get("cvssV3", {})
        score = cvss.get("baseScore", 5.0)
        severity = cvss.get("baseSeverity", "MEDIUM")
        return score, severity
    except Exception:
        return 5.0, "MEDIUM"

# Enrich node CVEs with real data
cve_scores = {}
for node,cve_list in node_cve_map.items():
    for cve_id in cve_list:
        cve_scores[cve_id] = fetch_cve_details(cve_id)

# -------------------
# Attack Graph Generation
# -------------------
# Example: create attack graph based on vulnerabilities and reachable nodes
AG = nx.DiGraph()
for node in nodes_with_cves:
    AG.add_node(node)
    for neighbor in G.neighbors(node):
        if neighbor in nodes_with_cves:
            AG.add_edge(node, neighbor)  # Possible lateral movement

# -------------------
# Objective Function (multi-criteria, e.g.: cost, risk, attack path length)
# -------------------
def objective_func(x):
    """
    x shape: (n_particles, len(nodes_with_cves))
    # Each: 1 = patch/mitigate, 0 = leave vulnerable
    Multi-criteria objectives:
        - Total cost (patching + potential loss)
        - Residual risk (unmitigated CVEs impact)
        - Shortest attack path risk
        - Computation time
    """
    costs = []
    risks = []
    attack_path_risks = []
    times = []
    for particle in x:
        start = time.time()
        cost = 0
        risk = 0
        # Patching, mitigation, cost
        patched_nodes = set()
        for idx, val in enumerate(particle):
            node = list(nodes_with_cves)[idx]
            cve_id = node_cve_map[node][0]
            score, severity = cve_scores.get(cve_id, (5.0, "MEDIUM"))
            patch_cost = score * 2  # Example cost formula
            attack_loss = score * 5  # Expected loss if attacked
            if val == 1:
                cost += patch_cost
                patched_nodes.add(node)
            else:
                cost += attack_loss
                risk += score
        # Attack path calculation: find shortest vulnerable path from entrypoint
        entry_nodes = [n for n in AG.nodes if AG.in_degree(n) == 0]
        min_path_risk = float('inf')
        for start_node in entry_nodes:
            for target_node in AG.nodes:
                try:
                    path = nx.shortest_path(AG, start_node, target_node)
                    path_risk = sum(
                        cve_scores.get(node_cve_map.get(n,[None])[0], (5.0,))[0]
                        for n in path if n not in patched_nodes
                    )
                    min_path_risk = min(min_path_risk, path_risk)
                except (nx.NetworkXNoPath, KeyError):
                    continue
        comp_time = time.time() - start
        costs.append(cost)
        risks.append(risk)
        attack_path_risks.append(min_path_risk)
        times.append(comp_time)
    # Weighted normalization (customizable)
    costs = np.array(costs)
    risks = np.array(risks)
    path_risks = np.array(attack_path_risks)
    times = np.array(times)
    costs_norm = (costs-costs.min())/(costs.max()-costs.min()+1e-8)
    risks_norm = (risks-risks.min())/(risks.max()-risks.min()+1e-8)
    path_risks_norm = (path_risks-path_risks.min())/(path_risks.max()-path_risks.min()+1e-8)
    times_norm = (times-times.min())/(times.max()-times.min()+1e-8)
    # Example weights: cost: 0.4, risk: 0.3, path risk: 0.2, time: 0.1
    return (
        0.4*costs_norm + 0.3*risks_norm +
        0.2*path_risks_norm + 0.1*times_norm
    )

# -------------------
# PSO Optimization
# -------------------
options = {'c1': 0.6, 'c2': 0.4, 'w': 0.8}
dim = len(nodes_with_cves)
bounds = (np.zeros(dim), np.ones(dim))  # Patch or not (+ maybe degree of patching)

optimizer = GlobalBestPSO(n_particles=30, dimensions=dim, options=options, bounds=bounds)
best_obj, best_pos = optimizer.optimize(objective_func, iters=50)

print("Optimal mitigation vector for vulnerable nodes:", best_pos)
print("Best combined cost-risk-path-time score:", best_obj)
