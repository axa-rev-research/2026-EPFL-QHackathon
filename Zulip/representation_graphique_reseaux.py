import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import json

# 1. Lecture des données
chemin_fichier = '/users/eleves-a/2024/max.anglade/QuantumInsuranceResourceAllocations/small_15/correlation_matrix.csv'
dossier_base = os.path.dirname(os.path.abspath(chemin_fichier))
df_corr = pd.read_csv(chemin_fichier, index_col=0)

# 2. Création du graphe
G = nx.Graph()
nodes = df_corr.index.tolist()
G.add_nodes_from(nodes)

# 3. Ajout des arêtes selon le seuil
threshold = 0.4
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        node1 = nodes[i]
        node2 = nodes[j]
        weight = df_corr.loc[node1, node2]
        if pd.notna(weight) and weight > threshold:
            G.add_edge(node1, node2, weight=weight)

# 4. Définition et filtrage des groupes (composantes connexes)
tous_les_groupes = list(nx.connected_components(G))

# On ne garde que les composantes avec 3 sommets ou plus
clusters_valides = [comp for comp in tous_les_groupes if len(comp) >= 3]

# Association des points à leur groupe pour la couleur
cluster_dict = {}
for cluster_id, comp in enumerate(clusters_valides):
    for node in comp:
        cluster_dict[node] = cluster_id

# Les points exclus (taille < 3) reçoivent l'ID -1 pour être coloriés de la même façon (bruit/isolés)
for node in G.nodes():
    if node not in cluster_dict:
        cluster_dict[node] = -1

node_colors = [cluster_dict[node] for node in G.nodes()]

# 5. Exportation des clusters valides dans un fichier JSON
def sauvegarder_clusters_json(clusters_list, dossier):
    groupes = {}
    for idx, comp in enumerate(clusters_list):
        groupes[str(idx)] = list(comp)
    
    chemin_json = os.path.join(dossier, 'groupes_points.json')
    with open(chemin_json, 'w') as f:
        json.dump(groupes, f, indent=4)

sauvegarder_clusters_json(clusters_valides, dossier_base)

# 6. Visualisation
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, k=0.2, seed=42)

# Utilisation de la nouvelle syntaxe pour le colormap
cmap_obj = plt.get_cmap('tab20')

# Tracé
nx.draw_networkx_nodes(G, 
                       pos, 
                       node_size=200, 
                       node_color=node_colors, 
                       cmap=cmap_obj, 
                       alpha=0.9, 
                       edgecolors='grey')

nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

# Utilisation de r"..." et \geq pour corriger l'erreur de parsing
plt.title(r"Graphe des clusters (Seuil > " + str(threshold) + r", Taille $\geq$ 3)", fontsize=16)

plt.axis('off')
plt.tight_layout()
chemin_graphe = os.path.join(dossier_base, 'clusters_graph.png')
plt.savefig(chemin_graphe, dpi=150, bbox_inches='tight')

# Tracé (les points isolés d'ID -1 auront tous la même couleur "par défaut" en bout de colormap)
nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors, cmap=plt.cm.get_cmap('tab20', len(clusters_valides) + 1), alpha=0.9, edgecolors='grey')
nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

plt.title(f"Graphe des clusters (Seuil > {threshold}, Taille $\ge$ 3)", fontsize=16)
plt.axis('off')
plt.tight_layout()
chemin_graphe = os.path.join(dossier_base, 'clusters_graph.png')
plt.savefig(chemin_graphe, dpi=150, bbox_inches='tight')