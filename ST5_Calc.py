## encoding: utf-8
import streamlit as st
import numpy as np
import cantera as ct
from scipy.optimize import minimize_scalar
from scipy import optimize
import pandas as pd

st.set_page_config(
    page_title="ST5 - Calculadora",
    layout="wide",
    )

class ST5_Calc:
    def __init__(self, **kargs):        
        self.driver = ct.Solution('./Data/o2h2_he.yaml')
        self.driven = ct.Solution('airNASA9.yaml')
        self.driven.X =  "N2:0.79,O2:0.21"
            
    def Driver(self, **kargs):  
        self.T4_i = kargs['T4_i']
        self.XHe  = kargs['XHe']

        self.xH2 = (2/3)*(1-self.XHe)
        self.xO2 = (1/2)*self.xH2

        self.p4_i = kargs['p4_i']
        self.driver.TPX = self.T4_i, self.p4_i, 'H2:'+str(self.xH2)+',O2:'+str(self.xO2)+',He:'+str(self.XHe)
        self.driver.equilibrate('UV',log_level=-99)  

        if kargs['p4_f']:
            self.p4_f = kargs['p4_f']
            self.T4_f = minimize_scalar(self.acha_T, bounds=(self.T4_i, 10000.0)).x

            self.driver.TPX = self.T4_i, self.p4_f, 'H2:'+str(self.xH2)+',O2:'+str(self.xO2)+',He:'+str(self.XHe)
            self.driver.equilibrate('UV')
    
            self.driver.TP = self.T4_f, self.p4_f
            self.driver.equilibrate('TP')
        else:
            self.p4_f = 0    
        
        self.T4 = self.driver.T
        self.p4 = self.driver.P
    
        self.g4 = self.driver.cp/self.driver.cv
        self.a4 = (self.g4*self.T4*ct.gas_constant/self.driver.mean_molecular_weight)**0.5 
        
        self.r4 = self.driver.density
        self.s4 = self.driver.entropy_mass
        self.h4 = self.driver.enthalpy_mass

    def acha_T(self, T):
        self.driver.TPX = self.T4_i, self.p4_i, 'H2:'+str(self.xH2)+',O2:'+str(self.xO2)+',He:'+str(self.XHe)
        self.driver.equilibrate('UV')
        densidade = self.driver.density
    
        self.driver.TP = T, self.p4_f
        self.driver.equilibrate('TP')
        return (self.driver.density - densidade)**2.0


    def Driven(self, **kargs):  
        self.p1 = kargs['p1']
        self.T1 = kargs['T1']
        self.driven.TP = self.T1, self.p1
        self.driven.equilibrate('TP')
        
        self.g1 = self.driven.cp/self.driven.cv
        self.a1 = (self.g1*self.T1*ct.gas_constant/self.driven.mean_molecular_weight)**0.5 
        
        self.r1 = self.driven.density
        self.s1 = self.driven.entropy_mass
        self.h1 = self.driven.enthalpy_mass
  
    def Calc_Ms(self,**kargs):  
        self.ef = kargs['Eficiencia']
        def res(Ms):
            p21  = ( 1+ (2*self.g1/(self.g1+1))*(Ms*Ms-1) )
            u2a1 = (1/(self.g1+1))*(Ms - 1./Ms)
            return ( (self.ef*self.p4/self.p1) - p21*( 1. - ( self.ef**((1-self.g4)/(2*self.g4)) )*(self.a1/self.a4)*(self.g4-1)*u2a1)**(-2*self.g4/(self.g4-1))  )**2.0

       
        _ = np.arange(1.1, 40, 0.1)
        Res = 1e10
        for i, v in enumerate(_):
            if res(v) < Res:
                self.Ms = _[i]
                Res = res(v)
        


    def Shock12(self,**kargs):
        self.driven.TP = self.T1, self.p1
        self.driven.equilibrate('TP')

        if kargs['Us'] :
            self.us = kargs['Us']
            self.flag_us = 1
                
        else:
            self.us = self.Ms*self.a1
            self.flag_us = 0

        try:
            r2 = self.r1*4.0
            for i in np.arange(20):                        #processo iterativo para encontrar condicoes apos onda 
                p2 = self.p1 +  self.r1*self.us*self.us*(1.0 - self.r1/r2        )  #de choque
                h2 = self.h1 + 0.5*self.us*self.us*(1.0 - (self.r1/r2)**2.0 )
    
                self.driven.HP = h2,p2
                self.driven.equilibrate('HP',solver='vcs',estimate_equil=1)
                r2 = self.driven.density
        except:
            print('Erro em Shock12')


        self.p2 = p2
        self.T2 = self.driven.T
        
        self.g2 = self.driven.cp/self.driven.cv
        self.a2 = (self.g2*self.T2*ct.gas_constant/self.driven.mean_molecular_weight)**0.5 
        
        self.r2 = self.driven.density
        self.s2 = self.driven.entropy_mass
        self.h2 = self.driven.enthalpy_mass

        self.u2 = self.us*(1.-self.r1/self.r2)

    def Shock25(self, **kargs):
        self.driven.TP = self.T2,self.p2
        self.driven.equilibrate('TP',log_level=-99)

        try:
            r5 = self.r2*2.0
            for i in np.arange(50):
                vr = self.u2/(r5/self.r2  -  1.)
                p5 = self.p2 +  self.r2*(vr + self.u2)*(vr + self.u2)*( 1. - self.r2/r5 )
                h5 = self.h2 + 0.5*(p5 - self.p2)*( 1./self.r2 + 1./r5 )
    
                self.driven.HP = h5,p5
                self.driven.equilibrate('HP',solver='gibbs',estimate_equil=1)
                r5 = self.driven.density
        except:
            print('Erro em Shock25')

        self.p5 = p5
        self.T5 = self.driven.T
        
        self.g5 = self.driven.cp/self.driven.cv
        self.a5 = (self.g5*self.T5*ct.gas_constant/self.driven.mean_molecular_weight)**0.5 
        
        self.r5 = self.driven.density
        self.s5 = self.driven.entropy_mass
        self.h5 = self.driven.enthalpy_mass

        self.vr = vr
    
        Mr = (self.vr + self.u2)/self.a2

        try:
            self.Shock5E(**kargs)
        except:
            print('Não foi informado p5 Medido.')

    def Shock5E(self, **kargs):

        self.pe = kargs['pe']
        self.driven.SP = self.s5, self.pe
        self.driven.equilibrate('SP')
        
        self.Te = self.driven.T
        
        self.ge = self.driven.cp/self.driven.cv
        self.ae = (self.g5*self.T5*ct.gas_constant/self.driven.mean_molecular_weight)**0.5 
        
        self.re = self.driven.density
        self.se = self.driven.entropy_mass
        self.he = self.driven.enthalpy_mass


    def Tabela(self):
        self.df = pd.DataFrame()
        self.df['Parametro'] = ['p (MPa)','T (K)','r (kg/m3)','Mach','s (kJ/K)','h (MJ/K/kg)','v (m/s)','a_s (m/s)','gamma']
        self.df.set_index('Parametro', inplace=True)

        self.df['Driver']  = np.array([self.p4/1e6,self.T4,self.r4,     0, self.s4/1e3, self.h4/1e6 ,     0.0, self.a4, self.g4]).astype(float).round(2)
        self.df['Driven']  = np.array([self.p1/1e6,self.T1,self.r1,     0, self.s1/1e3, self.h1/1e6 ,     0.0, self.a1, self.g1]).astype(float).round(3)
        self.df['Condição2'] = np.array([self.p2/1e6,self.T2,self.r2, self.u2/self.a2, self.s2/1e3, self.h2/1e6 , self.u2, self.a2, self.g2]).astype(float).round(2)
        self.df['Condição5'] = np.array([self.p5/1e6,self.T5,self.r5,     0, self.s5/1e3, self.h5/1e6 ,     0.0, self.a5, self.g5]).astype(float).round(2)
        try:
           self. df['CondiçãoEqI']  = np.array([self.pe/1e6,self.Te,self.re,     0, self.se/1e3, self.he/1e6 ,     0.0, self.ae, self.ge]).astype(float).round(2)
        except:
            pass


    def cs_body(self):
        
        col1, col2 = st.columns(2)
        
        #######################################
        # COLUMN 1
        #######################################
        a = 5
        col1.subheader('Driver')
        col1.code(f'''Pressão P4 informada de {self.p4_i/1e6} MPa a frio,\ntemperatura inicial de {self.T4_i} K e\nconcentração molar de Hélio de {self.XHe * 1e2} %.\nA composição complementar foi uma proporção\nestquiométrica de H2 e ar.''')
            
        if self.p4_f:
            col1.code(f'''A pressão P4 quente informada foi de {self.p4_f/1e6} Mpa''')

        col1.code(f'''Definida as informações de entrada, os cálculos levaram o gás de Driver à seguinte configuração:\npressão = {self.df['Driver'][0]} MPa\ntemperatura = {self.df['Driver'][1]} K\ndensidade = {self.df['Driver'][2]} kg/m3\nentropia = {self.df['Driver'][4]} kJ/K\nentalpia = {1e3*self.df['Driver'][5]} kJ/K/kg\nvel. do som = {self.df['Driver'][7]} m/s\ncp/cv = {self.df['Driver'][8]}''')

        
        col1.subheader('Driven')
        col1.code(f'''Ar atmosférico à pressão e temperatura iniciais\nde {self.p1/1e3} kPa e {self.T1} K. As informações complementares\nsão:\ndensidade = {self.df['Driven'][2]} kg/m3\nentropia = {self.df['Driven'][4]} kJ/K\nentalpia = {1e3*self.df['Driven'][5]} kJ/K/kg\nvel. do som = {self.df['Driven'][7]} m/s\ncp/cv = {self.df['Driven'][8]}''')

        col1.subheader('Onda de Choque Inicidente')
        col1.code(f'''As informações fornecida levaram a uma onda\nde choque primária de Ms de {round(self.Ms,3)}, correspondendo\numa velocidade aproximada de {int(self.us)} m/s.''')
        
        
        # Display media
        
        col1.subheader('Condições Chocadas')
        if self.flag_us :
            col1.code(f'''A partir de Us medido de {int(self.us)} m/s e das condições\niniciais do Driven, as seguintes condições chocadas\nforam calculadas:\npressão = {1e3*self.df['Condição2'][0]} kPa\ntemperatura = {self.df['Condição2'][1]} K\ndensidade = {self.df['Condição2'][2]} kg/m3\nentropia = {self.df['Condição2'][4]} kJ/K\nentalpia = {self.df['Condição2'][5]} MJ/K/kg\nvel. do som = {self.df['Condição2'][7]} m/s\ncp/cv = {self.df['Condição2'][8]}
            ''')
        else:
            col1.code(f'''A partir de Us calculado de {int(self.us)} m/s através da\nrelação P4/P1 e das condições iniciais do Driven, as\nseguintes condições chocadas foram calculadas:\npressão = {1e3*self.df['Condição2'][0]} kPa\ntemperatura = {self.df['Condição2'][1]} K\ndensidade = {self.df['Condição2'][2]} kg/m3\nentropia = {self.df['Condição2'][4]} kJ/K\nentalpia = {self.df['Condição2'][5]} MJ/K/kg\nvel. do som = {self.df['Condição2'][7]} m/s\ncp/cv = {self.df['Condição2'][8]} ''')
    
        #######################################
        # COLUMN 2
        #######################################
        
        # Display interactive widgets
        
        col2.subheader('Condições Refletidas')
        col2.code(f'''A condição refletida foi calculada a partir dos\nparâmetros informados até aqui, levando aos seguintes\nnúmeros:\npressão = {self.df['Condição5'][0]} MPa\ntemperatura = {self.df['Condição5'][1]} K\ndensidade = {self.df['Condição5'][2]} kg/m3\nentropia = {self.df['Condição5'][4]} kJ/K\nentalpia = {self.df['Condição5'][5]} MJ/K/kg\nvel. do som = {self.df['Condição5'][7]} m/s\ncp/cv = {self.df['Condição5'][8]}\nAqui não se usou P5 medido.''')

        if self.pe:
            col2.subheader('Condições de Equilíbrio')
            col2.code(f'''A partir da condição refletida e da pressão P5 de\n{self.pe/1e6} MPa medida, a condição de quilíbrio calculada\nfoi:\npressão = {self.df['CondiçãoEqI'][0]} MPa\ntemperatura = {self.df['CondiçãoEqI'][1]} K\ndensidade = {self.df['CondiçãoEqI'][2]} kg/m3\nentropia = {self.df['CondiçãoEqI'][4]} kJ/K\nentalpia = {self.df['CondiçãoEqI'][5]} MJ/K/kg\nvel. do som = {self.df['CondiçãoEqI'][7]} m/s\ncp/cv = {self.df['Condição5'][8]}''')        
        
        return None



 


with st.sidebar:
    st.header("Entradas")
  
    st.subheader("Driver ")     
    p4_i  =  st.number_input(label="Pressão Inicial, MPa:"   , value=2.5 , min_value=0.01, step=0.1)
    T4_i  =  st.number_input(label="Temperatura Inicial, K:" , value=300., min_value=0., step=1.)
    XHe   =  st.number_input(label="Concentração de Hélio, %", value=65.0, min_value=0.,max_value=100.0, step=1.)                        

    st.divider()
    st.subheader("Driven ") 
    p1  =  st.number_input(label="Pressão Inicial, kPa:"   , value=3.0 , min_value=1., step=0.1)
    T1  =  st.number_input(label="Temperatura Inicial, K:" , value=300., min_value=0.)

    st.divider()
    st.subheader("Fator de Ganho P4/P1 x Ms *")     
    eta =  st.number_input(label="Fator (0 a 1):" , value=1.00, min_value=0.1, step=0.05)   
    
    
    st.divider()
    st.subheader("Informações Pós-Disparo (preencher com 0 se não as tiver)")     
    p4_f  =  st.number_input(label="Pressão Final p4, MPa:"  , value=0.0, min_value=0., step=0.1)
    Us =  st.number_input(label="Vel. Onda de Choque Incidente, m/s:" , value=0.00, min_value=0.0, step=10.0)   
    pe =  st.number_input(label="Pressão p5, MPa:", value=0.00, min_value=0.0, step=1.0)   

    st.write('* o fator tem origem no fator de ganho apresentado para corrigir razões de áreas entre seções do driver e do driven; aqui se usa como fator de ajuste. ')

ST5 = ST5_Calc( )
ST5.Driver(p4_i=p4_i*1e6, p4_f=p4_f*1e6, T4_i=300.0, XHe=XHe/100.)
ST5.Driven(p1=p1*1e3, T1=T1)
ST5.Calc_Ms(Eficiencia=eta)
ST5.Shock12(Us=Us)
ST5.Shock25(pe=pe*1e6)
ST5.Tabela()
ST5.cs_body()

