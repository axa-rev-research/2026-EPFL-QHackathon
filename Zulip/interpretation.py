import json
import pandas as pd
import numpy as np

def verifier_et_interpreter(chemin_results, chemin_claims, chemin_groupes, budget, p):
    # 1. Chargement des données
    with open(chemin_results, 'r') as f:
        res = json.load(f)
    with open(chemin_groupes, 'r') as f:
        groupes = json.load(f)
        
    df = pd.read_csv(chemin_claims).sort_values("claim_id").reset_index(drop=True)
    claim_to_idx = {cid: i for i, cid in enumerate(df["claim_id"])}
    
    x = np.array(res["variables"]["x_i_claims"])
    y = np.array(res["variables"]["y_i_clusters"])
    z = np.array(res["variables"]["z_i_budget"])

    # 2. Calcul du Bénéfice (Formule initiale)
    # Ri = P_i * (v_i * M_i * p + C_i + M_i) - (C_i + 2 * M_i)
    Ri = df["P_i"] * (df["v_i"] * df["M_i"] * p + df["C_i"] + df["M_i"]) - (df["C_i"])
    benefice_total = np.sum(Ri * x)

    # 3. Calcul du Coût et Budget
    indices_selectionnes = np.where(x == 1)[0]
    cout_total = df.loc[indices_selectionnes, "C_i"].sum()

    # 4. Vérification de la cohérence des Clusters
    clusters_valides = True
    details_clusters = []
    
    for idx_y, (nom_cluster, ids_sinistres) in enumerate(groupes.items()):
        etat_y = y[idx_y]
        if etat_y == 1:
            # Vérifier si tous les x_j du cluster sont bien à 1
            indices_x = [claim_to_idx[cid] for cid in ids_sinistres if cid in claim_to_idx]
            tous_actifs = np.all(x[indices_x] == 1)
            
            if not tous_actifs:
                clusters_valides = False
                details_clusters.append(f"Cluster {nom_cluster} : ERREUR (y=1 mais certains x=0)")
            else:
                details_clusters.append(f"Cluster {nom_cluster} : OK (y=1, x=1)")
        else:
            details_clusters.append(f"Cluster {nom_cluster} : Inactif (y=0)")

    # 5. Affichage du Rapport
    print("="*45)
    print("       BILAN DE PERFORMANCE DU MODÈLE")
    print("="*45)
    print(f"Bénéfice Net Total (R_tot) : {benefice_total:.2f}")
    print(f"Coût des sinistres (C_tot) : {cout_total:.2f} / {budget}")
    
    print("-" * 25)
    if cout_total <= budget:
        print("✅ BUDGET : Respecté")
    else:
        print(f"❌ BUDGET : DÉPASSÉ de {cout_total - budget:.2f}")

    print("-" * 25)
    if clusters_valides:
        print("✅ LOGIQUE CLUSTERS : Valide")
    else:
        print("❌ LOGIQUE CLUSTERS : Incohérente")
    
    for detail in details_clusters:
        print(f"   -> {detail}")

    print("-" * 25)
    valeur_slack = sum(bit * (2**k) for k, bit in enumerate(z))
    ecart_reel = budget - cout_total
    print(f"Vérification Slack (z) : {valeur_slack}")
    print(f"Écart théorique (B-C)  : {ecart_reel:.2f}")
    print("="*45)

# Utilisation :
verifier_et_interpreter("/users/eleves-a/2024/rami.chagnaud/Documents/QuantumInsuranceResourceAllocations/logs/run_2026-02-28_21-53-29/resultats_optimisation.json", 
                        "/users/eleves-a/2024/rami.chagnaud/Documents/QuantumInsuranceResourceAllocations/datasets/tiny_database/tiny_claims.csv", 
                        "/users/eleves-a/2024/rami.chagnaud/Documents/QuantumInsuranceResourceAllocations/datasets/tiny_database/tiny_clusters.json", 
                        15.0, 5.0)