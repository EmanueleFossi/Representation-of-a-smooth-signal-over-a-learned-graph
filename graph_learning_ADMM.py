import numpy as np

def graph_learning_admm(Z, alpha, rho=1.0, max_iters=1000, tol=1e-4):
    N = Z.shape[0]
    
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    U = np.zeros((N, N)) # Moltiplicatore di Lagrange (dual variable)
    
    for c_iter in range(max_iters):

        A_old = A.copy()

        #Aggiornamento variabile primale X nel mio caso A
        Costante = B - U - (0.5 * Z) / rho  #derivata rispetto ad x del lagrangiano
        
        # Applico i vincoli
        A = 0.5 * (Costante + Costante.T)
        A[A < 0] = 0
        np.fill_diagonal(A, 0)
        
        # Aggioranmento variabile primale Z della regolarizzazione

        M = A + U
        for i in range(N):
            somma_m = np.sum(M[i, :])
            
            # Soluzione in forma chiusa del grado ottimo riga per riga 
            grado_ottimo = 0.5 * (somma_m + np.sqrt(somma_m**2 + (4 * alpha) / rho))
            
            # Ridistribuiamo il grado ottimizzato sui singoli elementi della riga di B
            if somma_m > 0:
                B[i, :] = M[i, :] * (grado_ottimo / somma_m)
            else:
                B[i, :] = grado_ottimo / N
                
        # Applicazione vincoli 
        B = 0.5 * (B + B.T)
        np.fill_diagonal(B, 0)
        
        # Aggiornamento variabile duale U
        U = U + A - B
        
       #Criterio di arresto
        if c_iter > 0:
            norma_A_old = np.linalg.norm(A_old, 'fro')
            if norma_A_old > 1e-8:
                residuo = np.linalg.norm(A - A_old, 'fro') / norma_A_old
            else:
                residuo = np.linalg.norm(A - A_old, 'fro')
                
            if residuo < tol:
                print(f"-> ADMM ha raggiunto la convergenza in {c_iter} iterazioni.")
                break
    else:
        print(f"-> ADMM ha terminato le {max_iters}.")
            
    return A