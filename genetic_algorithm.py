import  numpy as np
import matplotlib.pyplot as plt
import sys



ind_Size = int(sys.argv[1])
n_ind = int(sys.argv[2])
n_gen = int(sys.argv[3])
p_mut = float(sys.argv[4])
pob = 1
Max = 0

x = np.linspace(0, 2, 300)
fx = -(0.1 + np.power((1 - x), 2) - 0.1 * np.cos(6 * np.pi * (1 - x))) + 2
vector4 = []


def bernoulli(p):
    if np.random.random() < p:
        return 1  # Exito
    else:
        return 0  # Fracaso


def init_population(indSize, pob):
    PN = []
    for i in range(pob):
        DEC = np.around((np.random.random() * (2 -0) + 0), decimals = indSize-1)
        #print(DEC)
        #print((int(PN[0][0:2],2)),(int(PN[0][3:62],2)))
        PN.append(decimal_to_binary(DEC, indSize))

    return  PN

def decimal_to_binary(DEC, indSize):
    ind = [int(DEC), int((DEC - int(DEC)) * np.power(10, indSize - 1))]
    Bin = [np.binary_repr(ind[0], 2), np.binary_repr(ind[1], 60)]
    return Bin[0] + Bin[1]


def binary_to_decimal(PN, Size):
    Ind = []

    for i in PN:
        divisor = len(list(repr(int(i[3:62], 2))))
        Ind.append(int(i[0:2],2)+int(i[3:62],2)/np.power(10,divisor))

    return Ind


def crossover(I1,I2):
    pc = np.random.randint(0, 62)
    AI1 = list(I1)
    AI2 = list(I2)

    CROSS1 = AI1[pc:]
    CROSS2 = AI2[pc:]

    AI1[pc:] = CROSS2
    AI2[pc:] = CROSS1
    SI1, SI2 = "".join(AI1), "".join(AI2)
    return SI1, SI2


def mutar_individuo(I, p, indSize):
    if bernoulli(p):
        DEC = binary_to_decimal([I], indSize)[0]
        d = DEC - int(DEC)
        s = repr(int(d*np.power(10,indSize-1)))
        pc = np.random.randint(0,indSize-1)
        DECL = list(s)
        DECL[pc]  = np.random.choice(['0','1','2','3','4','5','6','7','8','9'])
        NUM = int(DEC) + int("".join(DECL))/np.power(10,indSize-1)
        return decimal_to_binary(NUM,indSize)
    return I


def fitness_poblacion(P):
    FP = []
    for pi in P:
        FP.append(pi)

    return FP


def evaluate_poblacion(FP, size):
    DP = binary_to_decimal(FP[:], size)
    EP = []
    fxe = []
    for i in DP:
        fxe.append(-(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2)

    for i in fxe:
        EP.append(i/np.max(fxe))
    #EP.append(similarity(reference,i))
    return EP


def commulative_fitness(EP):
    totalFitness = np.sum(EP)
    relativeFitness = EP / totalFitness
    #print(EP)
    #print(relativeFitness)
    CF = [relativeFitness[0]]
    for i in range(1, len(relativeFitness)):
        CF.append(relativeFitness[i] + CF[i - 1])
    return CF


def rulette_selection(CF):
    ps = np.random.random()
    seleccion = 0
    for pi in CF:
        if pi > ps:
            return seleccion

        seleccion += 1


def mean_population(Pob, indSize):
    D = binary_to_decimal(Pob, indSize)
    fxe =[]
    for i in D:
        fxe.append(-(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2)

    return np.sum(fxe) / len(Pob)


def main():
    """
    Uso: python agOneMax.py popSize indSize mutation steps
    popSize: size of the population
    indSize: size of the individuals
    mutations: mutation rate
    steps: evolution steps
    """
    popSize = n_ind  # int(args[0])
    indSize = ind_Size # int(args[1])
    mutation = p_mut#float(args[2])
    steps = n_gen# int(args[3]) # generations

    stepsPlot = np.floor(steps / 4)

    mP = []

    Pob = init_population(indSize, popSize)
    Max = []
    #print(Pob)

    for t in range(steps):

        Max.append(np.max(binary_to_decimal(Pob, indSize)))

        EP = evaluate_poblacion(Pob, indSize)

        CF = commulative_fitness(EP)

        PN = Pob[:]
        for i in range(popSize):
            if bernoulli(0.85):
                inds = [rulette_selection(CF) for i in range(2)]
                H1, H2 = crossover(Pob[inds[0]], Pob[inds[1]])
                PN[inds[0]] = H1
                PN[inds[1]] = H2

            #print(binary_to_decimal([H1], indSize))
        for pi in range(popSize):
            PN[pi] = mutar_individuo(PN[pi], mutation, indSize)

        Pob = PN[:]
        mP.append(mean_population(Pob, indSize))
        #print(Pob)
        if t % stepsPlot == 0:
            vector4.append([binary_to_decimal(Pob,indSize), str(t)])

    plt.figure(1,figsize=(20,10))


    plt.subplot(241)
    plt.title("Generación " + '0',fontsize = 9 )
    for i in vector4[0][0]:
        fxe = -(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2
        plt.plot(i, fxe, 'k+')
    plt.plot(x, fx)

    plt.subplot(242)
    plt.title("Generación " + vector4[1][1], fontsize = 9 )

    for i in vector4[1][0]:
        fxe = -(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2
        plt.plot(i, fxe, 'k+')
    plt.plot(x, fx)

    plt.subplot(243)
    plt.title("Generación " + vector4[2][1],fontsize = 9 )

    for i in vector4[2][0]:
        fxe = -(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2
        plt.plot(i, fxe, 'k+')
    plt.plot(x, fx)

    plt.subplot(244)
    plt.title("Generación " + vector4[3][1], fontsize = 9 )

    plt.subplot
    for i in vector4[3][0]:
        fxe = -(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2
        plt.plot(i, fxe, 'k+')
    plt.plot(x, fx)

    plt.subplot(223)
    plt.title("Promedio de f(x)" , fontsize = 9 )
    plt.plot(range(steps), mP)

    plt.subplot(224)
    plt.title("Generación Final" , fontsize = 9 )
    fxe = []
    xv = binary_to_decimal(Pob, indSize)
    for i in xv:
        fxe.append(-(0.1 + np.power((1 - i), 2) - 0.1 * np.cos(6 * np.pi * (1 - i))) + 2)
    plt.plot(xv, fxe, 'k+')
    plt.plot(x, fx)
    max = np.max(Max)
    print("Mejor individuo: " + str(max))


    #plt.ylim(0,3)
    plt.show()


if __name__ == "__main__":
    main()


