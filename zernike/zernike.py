import numpy as np 
import math
from scipy.special import (factorial, comb as nchoosek)
from collections import defaultdict

class Zernike():
    def __init__(self):
        
        self.binomial_dict = {}
        self.factorial_dict = {}
        self.q_klv_dict = {}
        self.clm_dict = {}
        
        self.moments = None
        self.points = None
        self.logger = []
    
    
    def get_nlms(self, order):
        nlms = []
        
        for n in range(order+1):
            for l in range(n+1):
                for m in range(l+1):
                    if (n-l)%2 == 0:
                        nlms.append([n,l,m])
                            
        return np.array(nlms)
    
    def binomial(self, n,k):
        
        if (n,k) in self.binomial_dict.keys():
            output = self.binomial_dict[(n,k)]
        else:
            output = nchoosek(n,k)
            self.binomial_dict[(n,k)] = output

        return output
    
    def fact(self, n):
        if n in self.factorial_dict.keys():
            output = self.factorial_dict[n]
        else:
            output = factorial(n)
            self.factorial_dict[n] = output

        return output
    
    def chi_nlm_rst(self, n,l,m, mode):
        output = self.c_l_m(l,m) * (2.**(-m)) + 0j
        k=(n-l)//2
            
        output = output * self.sum1(n,l,m,k, mode)
        
        if mode == 'omega':
            output = np.conj(output) * 0.75 / np.pi

        return output
    
    def c_l_m(self,l,m):

        if (l,m) in self.clm_dict.keys():
            output = self.clm_dict[(l,m)]
        else:
            output = (2*l + 1) * self.fact(l+m) * self.fact(l-m)
            output = math.sqrt(output)
            output /= self.fact(l)

            self.clm_dict[(l,m)] = output

        return output
    
    def sum1(self, n, l, m, k, mode):
        output = 0 + 0j

        for nu in range(k+1):
            output += self.Q_klv(k, l, nu) * self.sum2(n, l, m, k, nu, mode)

        return output
    
    def Q_klv(self, k, l, nu):
        if (k,l,nu) in self.q_klv_dict.keys():
            output = self.q_klv_dict[(k,l,nu)]

        else:
            output = (-1)**(k+nu)
            output /= 2.**(2*k)
            output *= math.sqrt((2*l + 4*k + 3)/3)
            output *= self.binomial(2*k, k)
            output *= self.binomial(k, nu)
            output *= self.binomial(2*(k+l+nu)+1, 2*k)
            output /= self.binomial(k+l+nu, k)

            self.q_klv_dict[(k,l,nu)] = output

        return output
    
    def sum2(self, n, l, m, k, nu, mode):
        output = 0 + 0j

        for alpha in range(nu+1):
            output += self.binomial(nu, alpha) * self.sum3(n, l, m, k, nu, alpha, mode)

        return output
    
    
    def sum3(self, n, l, m, k, nu, alpha, mode):
        output = 0 + 0j

        for beta in range(nu-alpha+1):
            output += self.binomial(nu-alpha, beta) * self.sum4(n, l, m, k, nu, alpha, beta, mode)

        return output
    
    def sum4(self, n, l, m, k, nu, alpha, beta, mode):

        output = 0 + 0j

        for u in range(m+1):
            output += ((-1.)**(m-u)) * self.binomial(m, u) * ((0 + 1j)**u) * self.sum5(
                                                        n, l, m, nu, alpha, beta, u, mode)

        output=output.real-(output.imag)*(0+1j)
        return output
    
    def sum5(self, n, l, m, nu, alpha, beta, u, mode):

        output = 0 + 0j

        for mu in range(((l-m)//2)+1):
            temp = ((-1.)**mu) * (2.**(-2*mu)) 
            temp *= self.binomial(l, mu) 
            temp *= self.binomial(l-mu, m+mu) 
            temp *= self.sum6(n, l, m, nu, alpha, beta, u, mu, mode)
            output += temp

        return output
    

    def sum6(self, n, l, m, nu, alpha, beta, u, mu, mode):

        output = 0 + 0j

        for v in range(mu+1):

            r=2*(v+alpha)+u
            s=2*(mu-v+beta)+m-u
            t=2*(nu-alpha-beta-mu)+l-m
            
            temp = self.binomial(mu, v)
            
            if mode == 'omega':
                temp *= self.moments[r,s,t]
                
            elif mode == 'zernike':

                x = self.points[:,0]
                y = self.points[:,1]
                z = self.points[:,2]
                
                factor = (x**r) * (y**s) * (z**t)
                self.logger.append([(r,s,t),factor])
                
                temp *= factor
                
            else:
                # chi computation
                pass
            
            output += temp

        return output

    def zernike_zernike_function(self, points, order):
        self.points = points
        output = self.calculate_descriptors(order, mode='zernike')
        self.points = None 
        return output 

    def calculate_chi_nlm(self, order):
        output = self.calculate_descriptors(order, mode='chi')
        return output 

    def calculate_zernike_moments(self, moments, order):
        self.moments = moments 
        output = self.calculate_descriptors(order, mode='omega')
        self.moments = None 
        return output 
    
    def calculate_descriptors(self, order, mode):
        # mode == 'zernike' compute Z_nlm
        # mode == 'chi' compute chi_nlm_rst
        # mode == 'omega' compute omega_nlm
        
        self.q_klv_dict = {}
        self.clm_dict = {}
        
        nlms = self.get_nlms(order)

        chi_d = {}

        for triplet in nlms:
            n,l,m = triplet

            chi = self.chi_nlm_rst(n,l,m, mode)
            
            chi_d[(n,l,m)] = chi

            if m>0:
                
                if mode == 'omega':
                    neg_chi = ((-1)**m) * np.conj(chi)
                else:
                    neg_chi = self.chi_nlm_rst(n,l,-m, mode)
                
                chi_d[(n,l,-m)] = neg_chi
                                    
        return chi_d
    
    def clear_dicts(self):
        self.binomial_dict = {}
        self.factorial_dict = {}
        self.q_klv_dict = {}
        self.clm_dict = {}
        self.logger = []
        