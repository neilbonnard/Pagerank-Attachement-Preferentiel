import random
import bisect

#Met le graphe dans une liste d'adjacence
def read_graph(filename):
    with open(filename, "r") as f:
        n = int(f.readline().strip())
        m = int(f.readline().strip())

        graph = [[] for _ in range(n + 1)]  # graph[0] inutilisé

        for _ in range(n):
            parts = f.readline().split()
            u = int(parts[0])
            nb_neighbors = int(parts[1])

            idx = 2
            for _ in range(nb_neighbors):
                v = int(parts[idx])
                graph[u].append(v)
                idx += 2  # on saute aussi le poids

    return graph

#Calcule les degrés entrants à partir du graphe
def compute_in_degrees_from_graph(graph):
    n = len(graph) - 1
    in_degree = [0] * (n + 1)

    for u in range(1, n + 1):
        for v in graph[u]:
            in_degree[v] += 1

    return in_degree

#Choisit des cibles pour les nouveaux liens en utilisant la méthode de l'attachement préférentiel
def choose_preferential_targets(in_degree, nb_links):
    n = len(in_degree) - 1

    if nb_links > n:
        raise ValueError("nb_links ne peut pas dépasser le nombre de sommets existants")

    cumulative = []
    total = 0

    for i in range(1, n + 1):
        total += in_degree[i] + 1
        cumulative.append(total)

    targets = set()

    while len(targets) < nb_links:
        r = random.uniform(0, total)
        idx = bisect.bisect_left(cumulative, r)
        targets.add(idx + 1)

    return list(targets)

#Ajoute des sommets avec des liens préférentiels au graphe
def add_preferential_nodes(graph, nb_new_nodes, nb_links_per_new_node):
    old_n = len(graph) - 1
    in_degree = compute_in_degrees_from_graph(graph)

    for new_node in range(old_n + 1, old_n + nb_new_nodes + 1):
        targets = choose_preferential_targets(in_degree, nb_links_per_new_node)

        graph.append(targets)

        # le nouveau sommet a pour l’instant degré entrant 0
        in_degree.append(0)

        # les anciennes cibles gagnent un lien entrant
        for v in targets:
            in_degree[v] += 1

    return graph

#Ajoute des sommets isolés au graphe
def add_isolated_nodes(graph, nb_new_nodes):
    for _ in range(nb_new_nodes):
        graph.append([])

    return graph

#Écrit le graphe dans un fichier au format texte
def write_graph(graph, filename):
    n = len(graph) - 1
    m = sum(len(graph[u]) for u in range(1, n + 1))

    with open(filename, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{m}\n")

        for u in range(1, n + 1):
            neighbors = graph[u]
            degree = len(neighbors)

            f.write(f"{u} {degree}")

            if degree > 0:
                weight = 1.0 / degree
                for v in neighbors:
                    f.write(f" {v} {weight:.12f}")

            f.write("\n")