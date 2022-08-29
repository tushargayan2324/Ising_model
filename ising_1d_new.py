import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit(nopython=True)
def energy(M): #energy for M array of sites 
    J = 1.0 #constant
    N = len(M)
    E_array = np.array([M[i%N]*M[(i+1)%N] for i in range(N)]) #two neighbours interacting spins
    #E_array = np.array([M[i]*M[(i+1)%N]*M[(i+2)%N] for i in range(N-2)])    
    #E_array = np.array( [ M[i%N]*(M[(i+1)%N]+M[(i-1)%N]) for i in range(N)] )
    return -J*(np.sum(E_array))

@numba.jit(nopython=True)
def cal_mag(M):
    return abs(np.sum(M))/len(M)


@numba.jit(nopython=True)
def main(T):
    L = 10**4 #size of the lattice
    time_steps = 10**5 ##Time steps
    #print(L,time_steps)
    print(T)
    spin_values = np.array([1,-1])
    M = np.random.choice(spin_values,size=L)
    N = len(M)
    #mag = 0 #cal_mag(M)
    
    Erg_array_time_step = np.zeros(time_steps)
    #Erg_array_temp =  np.zeros(len(temp_array))

    Mag_array = np.zeros(time_steps)
    
    b = float(1)/T #b depends on temperature

    for i in range(time_steps):

        a = np.random.randint(0,L)
        
        #mag = cal_mag(M)
        #erg = energy(M)

        M[a] = (-1)*M[a]

        #del_erg = float(energy(M) - erg) #new energy - erg old
        del_erg = 2*M[a%N]*(M[(a+1)%N]+ M[(a-1)%N]) #*M[(a+2)%N]
        del_mag = -2*M[a]

        if del_erg < 0:
            Erg_array_time_step[i] += Erg_array_time_step[i-1] + del_erg
            Mag_array[i] += Mag_array[i-1] + del_mag

        elif del_erg > 0:
            if np.exp(-b*(del_erg)) > np.random.random():
                Erg_array_time_step[i] += Erg_array_time_step[i-1] + del_erg
                Mag_array[i] += Mag_array[i-1] + del_mag
            else:
                M[a] = (-1)*M[a]
                Erg_array_time_step[i] = Erg_array_time_step[i-1]
                Mag_array[i] = Mag_array[i-1]
        elif del_erg ==0:
            M[a] = (-1)*M[a]
            Erg_array_time_step[i] = Erg_array_time_step[i-1]
            Mag_array[i] = Mag_array[i-1]

    Erg_eq = Erg_array_time_step[-1]/time_steps
    Mag_eq = Mag_array[-1]/time_steps
    
    return Erg_array_time_step, Mag_array, Erg_eq , Mag_eq


Pts = 150 #no. of pts on Energy-Temp graph

temp_array = np.linspace(0.001,5,Pts)
Erg_Array = np.zeros(Pts)
Mag_array = np.zeros(Pts)



# for i in range(len(temp_array)):
#     T = main(temp_array[i])
#     Erg_Array[i] = T[1]
#     Mag_array[i] = T[2]


time_steps = 10**5
plt.plot(np.linspace(0,time_steps,time_steps), main(1)[1])

plt.plot(temp_array,Erg_Array)
#plt.scatter(temp_array,Erg_Array)
plt.plot(temp_array,Mag_array)


plt.show()
