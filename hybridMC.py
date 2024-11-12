num_of_over_relax = 4
def hybrid_Monte_Carlo(num_of_over_relax, kagome): 
    CanoMC(kagome)
    for k in range(num_of_over_relax):
        Over_relaxation(kagome)
