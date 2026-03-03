import numpy as np

def kmedoids(B, k, max_iter=100, n_init=10, random_state=None):
    """
    
    Paramètres
    ----------
    B          : np.ndarray de shape (N, N) — matrice de distances symétrique
    k          : int — nombre de clusters
    max_iter   : int — nombre max d'itérations par run
    n_init     : int — nombre de runs avec initialisation aléatoire
    random_state : int ou None

    Retourne
    --------
    labels     : np.ndarray (N,) — label de cluster pour chaque point
    medoids    : np.ndarray (k,) — indices des medoïdes finaux
    cost       : float — coût total (somme des distances aux medoïdes)
    """
    rng = np.random.default_rng(random_state)
    N = B.shape[0]
    assert B.shape == (N, N), "B doit être carrée"
    assert k <= N, "k ne peut pas dépasser N"

    best_labels, best_medoids, best_cost = None, None, np.inf

    for _ in range(n_init):
        medoids = rng.choice(N, size=k, replace=False)

        for iteration in range(max_iter):
            labels = np.argmin(B[:, medoids], axis=1) 


            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                sub_matrix = B[np.ix_(members, members)]
                #C'est la sous-matrice des distances entre tous les membres du cluster.
                best_local = np.argmin(sub_matrix.sum(axis=1))
                #somme sur les colonnes → une valeur par ligne = somme des distances de chaque membre vers tous les autres :
                new_medoids[c] = members[best_local]
                # donc ex si best local =1 c'est le vrai medoid 

            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                break
            medoids = new_medoids

        labels = np.argmin(B[:, medoids], axis=1)
        cost = sum(B[i, medoids[labels[i]]] for i in range(N))

        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
            best_medoids = medoids.copy()

    return best_labels, best_medoids, best_cost


# ─── Exemple d'utilisation ───────────────────────────────────────────────────

if __name__ == "__main__":
    N = 20   # nombre de points
    k = 3    # nombre de clusters

    # Simulation d'une matrice de distances symétrique
    rng = np.random.default_rng(42)
    coords = rng.random((N, 2)) * 100          # coordonnées fictives 2D
    from scipy.spatial.distance import cdist
    B = cdist(coords, coords)                   # matrice N×N

    labels, medoids, cost = kmedoids(B, k=k, n_init=10, random_state=42)

    print(f"Medoïdes (indices) : {medoids}")
    print(f"Coût total         : {cost:.4f}")
    for c in range(k):
        members = np.where(labels == c)[0].tolist()
        print(f"  Cluster {c} (medoïde={medoids[c]}) : {members}")