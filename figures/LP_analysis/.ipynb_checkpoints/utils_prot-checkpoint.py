import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit,set_num_threads,prange

Ncont =28
Nstruct =10000
Ninterface = 144
Ncontface=9

s1=624
s2 = 6682
sface=0

Cint = np.loadtxt("./contact_interface_144_new.csv",dtype="int",delimiter=',')
Cs = np.loadtxt("./contact_maps_10000.dat",dtype="int")

#code association number to a.a

code=np.zeros(20, dtype="object")

code[0]='C' #Cys
code[1]='M' #Met 
code[2]='F' #Phe 
code[3]='I' #Ile 
code[4]='L' #Leu
code[5]='V' #Val
code[6]='W' #Trp
code[7]='Y' #Tyr
code[8]='A' #Ala
code[9]='G' #Gly 
code[10]='T' #Thr
code[11]='S' #Ser 
code[12]='N' #Asn
code[13]='Q' #Gln
code[14]='D' #Asp
code[15]='E' #Glu 
code[16]='H' #His
code[17]='R' #Arg
code[18]='K' #Lys
code[19]='P' #Pro

#code association a.a to number
Dict = {"C": 0, 'M': 1, "F": 2, "I": 3, "L": 4, "V": 5, "W": 6, "Y": 7, "A": 8, "G": 9, "T": 10, "S": 11, "N": 12, "Q": 13, "D": 14, "E": 15, "H": 16, "R": 17, "K": 18, "P": 19 }

#           Cys,   Met,   Phe,   Ile,   Leu,   Val,   Trp,   Tyr,   Ala,   Gly,   Thr,   Ser,   Asn,   Gln,   Asp,   Glu,   His,   Arg,   Lys,   Pro

list_en = [[ -5.44, -4.99, -5.80, -5.50, -5.83, -4.96, -4.95, -4.16, -3.57, -3.16, -3.11, -2.86, -2.59, -2.85, -2.41, -2.27, -3.60, -2.57, -1.95, -3.07], # Cys
        [ -4.99, -5.46, -6.56, -6.02, -6.41, -5.32, -5.55, -4.91, -3.94, -3.39, -3.51, -3.03, -2.95, -3.30, -2.57, -2.89, -3.98, -3.12, -2.48, -3.45], # Met
        [ -5.80, -6.56, -7.26, -6.84, -7.28, -6.29, -6.16, -5.66, -4.81, -4.13, -4.28, -4.02, -3.75, -4.10, -3.48, -3.56, -4.77, -3.98, -3.36, -4.25], # Phe
        [ -5.50, -6.02, -6.84, -6.54, -7.04, -6.05, -5.78, -5.25, -4.58, -3.78, -4.03, -3.52, -3.24, -3.67, -3.17, -3.27, -4.14, -3.63, -3.01, -3.76], # Ile
        [ -5.83, -6.41, -7.28, -7.04, -7.37, -6.48, -6.14, -5.67, -4.91, -4.16, -4.34, -3.92, -3.74, -4.04, -3.40, -3.59, -4.54, -4.03, -3.37, -4.20], # Leu
        [ -4.96, -5.32, -6.29, -6.05, -6.48, -5.52, -5.18, -4.62, -4.04, -3.38, -3.46, -3.05, -2.83, -3.07, -2.48, -2.67, -3.58, -3.07, -2.49, -3.32], # Val
        [ -4.95, -5.55, -6.16, -5.78, -6.14, -5.18, -5.06, -4.66, -3.82, -3.42, -3.22, -2.99, -3.07, -3.11, -2.84, -2.99, -3.98, -3.41, -2.69, -3.73], # Trp
        [ -4.16, -4.91, -5.66, -5.25, -5.67, -4.62, -4.66, -4.17, -3.36, -3.01, -3.01, -2.78, -2.76, -2.97, -2.76, -2.79, -3.52, -3.16, -2.60, -3.19], # Tyr
        [ -3.57, -3.94, -4.81, -4.58, -4.91, -4.04, -3.82, -3.36, -2.72, -2.31, -2.32, -2.01, -1.84, -1.89, -1.70, -1.51, -2.41, -1.83, -1.31, -2.03], # Ala
        [ -3.16, -3.39, -4.13, -3.78, -4.16, -3.38, -3.42, -3.01, -2.31, -2.24, -2.08, -1.82, -1.74, -1.66, -1.59, -1.22, -2.15, -1.72, -1.15, -1.87], # Gly
        [ -3.11, -3.51, -4.28, -4.03, -4.34, -3.46, -3.22, -3.01, -2.32, -2.08, -2.12, -1.96, -1.88, -1.90, -1.80, -1.74, -2.42, -1.90, -1.31, -1.90], # Thr
        [ -2.86, -3.03, -4.02, -3.52, -3.92, -3.05, -2.99, -2.78, -2.01, -1.82, -1.96, -1.67, -1.58, -1.49, -1.63, -1.48, -2.11, -1.62, -1.05, -1.57], # Ser
        [ -2.59, -2.95, -3.75, -3.24, -3.74, -2.83, -3.07, -2.76, -1.84, -1.74, -1.88, -1.58, -1.68, -1.71, -1.68, -1.51, -2.08, -1.64, -1.21, -1.53], # Asn
        [ -2.85, -3.30, -4.10, -3.67, -4.04, -3.07, -3.11, -2.97, -1.89, -1.66, -1.90, -1.49, -1.71, -1.54, -1.46, -1.42, -1.98, -1.80, -1.29, -1.73], # Gln
        [ -2.41, -2.57, -3.48, -3.17, -3.40, -2.48, -2.84, -2.76, -1.70, -1.59, -1.80, -1.63, -1.68, -1.46, -1.21, -1.02, -2.32, -2.29, -1.68, -1.33], # Asp
        [ -2.27, -2.89, -3.56, -3.27, -3.59, -2.67, -2.99, -2.79, -1.51, -1.22, -1.74, -1.48, -1.51, -1.42, -1.02, -0.91, -2.15, -2.27, -1.80, -1.26], # Glu
        [ -3.60, -3.98, -4.77, -4.14, -4.54, -3.58, -3.98, -3.52, -2.41, -2.15, -2.42, -2.11, -2.08, -1.98, -2.32, -2.15, -3.05, -2.16, -1.35, -2.25], # His
        [ -2.57, -3.12, -3.98, -3.63, -4.03, -3.07, -3.41, -3.16, -1.83, -1.72, -1.90, -1.62, -1.64, -1.80, -2.29, -2.27, -2.16, -1.55, -0.59, -1.70], # Arg
        [ -1.95, -2.48, -3.36, -3.01, -3.37, -2.49, -2.69, -2.60, -1.31, -1.15, -1.31, -1.05, -1.21, -1.29, -1.68, -1.80, -1.35, -0.59, -0.12, -0.97], # Lys
        [ -3.07, -3.45, -4.25, -3.76, -4.20, -3.32, -3.73, -3.19, -2.03, -1.87, -1.90, -1.57, -1.53, -1.73, -1.33, -1.26, -2.25, -1.70, -0.97, -1.75]] # Pro

imat=np.array(list_en)

def sqntol(sequence):                     #convert sequence number to amminoac
    str1= ""
    coding=np.zeros(27,dtype="object")
    for i in range(27):
        coding[i]=code[sequence[i]]

    return (str1.join(coding))

def sqlton(sequence):                    #convert sequence amminoac to letters
    arr=list(sequence)
    coding=np.zeros(27, dtype="int")
    for i in range(27):
        coding[i] = Dict[arr[i]]

    return (coding)

@jit(nopython=True)
def single_contact(idn,k,sequence):
    val = imat[sequence[(Cs[(k*Ncont)+idn][1]-1)]][sequence[(Cs[(k*Ncont)+idn][2]-1)]]   
    return val
    
@jit(nopython=True)
def energy(k, sequence):
    en=0.0
    for i in range(Ncont):
        en += imat[sequence[(Cs[(k*Ncont)+i][1]-1)]][sequence[(Cs[(k*Ncont)+i][2]-1)]]
       
    return (en)

@jit(nopython=True)
def energybindtrue(f,sequence_a,sequence_b):
    en =0.0
    
    for i in range(Ncontface):
        en += imat[sequence_a[(Cint[(f*Ncontface)+i][1]-1)]][sequence_b[(Cint[(f*Ncontface)+i][2]-1)]]
         
    return (en)

@jit(nopython=True)
def pnat(s, sequence):
    normalize =0.0
    for k in range(Nstruct):
        normalize += np.exp(-energy(k,sequence))
    
    pnat = np.exp(-energy(s,sequence))/normalize
    
    return pnat

@jit(nopython=True)
def pbind(sface,sequence_a,sequence_b):
    normalize=0.0
    for k in range(Ninterface):
        normalize += np.exp(-energybindtrue(k,sequence_a,sequence_b))
        
    pbind=np.exp(-energybindtrue(sface,sequence_a,sequence_b))/normalize
    
    return pbind

@jit(nopython=True)
def modpbind(sface,sequence_a,sequence_b):
        
    pbind=np.exp(-energybindtrue(sface,sequence_a,sequence_b))
    
    return pbind