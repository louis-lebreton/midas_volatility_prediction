import numpy as np
import pandas as pd
from scipy.stats import norm
import math

class RealizedVolatilityMeasure:
    """Classe qui calcule des mesures de volatilité sur les data "5-min crude oil E-Mini futures"
        - calcule le return
        - calcule realized volatility (RV)
        - calcule bi-power variation (BPV)
        - calcule tri-power quarticity (TQt)
        - test Z-ratio pour détecter les jumps
        - calcule la composante continue (CRV)
        - calcule la composante jump (CJ)
        - calcule semi-variances + et -
        - génère un df avec les mesures de volatilité
    """
    
    def __init__(self, high_freq_data):
        """
        Parameters:
        high_freq_data (pd.DataFrame): données haute fréquence 5-min crude oil E-Mini futures
        """
        self.data = high_freq_data
        self.data_daily = None
        self.u1 = 0.7979  # pour calculer la bi-power variation
    
    def calculate_returns(self, price_col='price'):
        """
        calcule les rendements logarithmiques
        
        Parameters:
        price_col (str): nom de la colonne contenant les prix
        
        Returns:
        pd.DataFrame: df contenant les rendements calculés
        """
        # log (prix t / prix t-1)
        self.data['return'] = np.log(self.data[price_col] / self.data[price_col].shift(1))
        
        return self.data
    
    def realized_volatility(self):
        """
        calcule realized volatility (RV)
        
        Returns:
        pd.DataFrame: df contenant la realized volatility (RV)
        """
        # rv par jour
        rv_by_month = self.data.groupby(pd.Grouper(freq='ME'))['return'].apply(lambda x: np.sum(x**2)) # à modifier ----------------------------
        # data par jour
        self.data_daily = rv_by_month.to_frame(name='rv')
        
        return self.data_daily
    
    def semi_variance(self):
        """
        calcule les semi-variances positives (RVt+) et négatives (RVt-)
        
        Returns:
        pd.DataFrame: df contenant RVt+ et RVt-
        """
        # RVt+
        rv_pos_by_day = self.data.groupby(pd.Grouper(freq='ME')).apply(  # ------------------------------------------------
            lambda x: np.sum(x.loc[x['return'] > 0, 'return']**2)
        )
        
        # RVt-
        rv_neg_by_day = self.data.groupby(pd.Grouper(freq='ME')).apply( # ------------------------------------------------
            lambda x: np.sum(x.loc[x['return'] < 0, 'return']**2)
        )
        
        semi_var_data = pd.DataFrame({
            'rv_pos': rv_pos_by_day,
            'rv_neg': rv_neg_by_day
        })
    
        self.data_daily = pd.merge(self.data_daily, semi_var_data, 
                                    left_index=True, right_index=True)
    
        return self.data_daily
    
    def bipower_variation(self):
        """
        calcule variation bi-power (BPV)
            
        Returns:
        pd.DataFrame: df contenant la variation bi-power (BPV)
        """
        # constante à appliquer
        u1_squared_inv = 1 / (self.u1 ** 2)
        
        # somme de returns absolus
        bpv_by_month = self.data.groupby(pd.Grouper(freq='ME'))['return'].apply( # à modifier ----------------------------------------------------
            lambda x: u1_squared_inv * np.sum(np.abs(x.values[1:]) * np.abs(x.values[:-1]))
        )
        
        data_bpv = bpv_by_month.to_frame(name='bpv')
        # ajout au data_daily
        self.data_daily = pd.merge(self.data_daily, data_bpv, 
                                  left_index=True, right_index=True)
        
        return self.data_daily
    
    def realized_tripower_quarticity(self):
        """
        calcule la quarticity tri-power réalisée (TQt) nécessaire pour le test Z-ratio
            
        Returns:
        pd.DataFrame: df contenant la quarticity tri-power (TQ)
        """
        # constante
        u_4_3 = 2**(2/3) * math.gamma(7/6) / math.gamma(1/2)
        const = 1 / (u_4_3 ** 3)
        
        tq_by_month = self.data.groupby(pd.Grouper(freq='ME'))['return'].apply( # à modifier ----------------------------------------------------
            lambda x: const * np.sum(
                np.abs(x.values[:-2])**(4/3) * 
                np.abs(x.values[1:-1])**(4/3) * 
                np.abs(x.values[2:])**(4/3)
            )
        )
        
        data_tq = tq_by_month.to_frame(name='tq')
        
        # ajout au data_daily
        self.data_daily = pd.merge(self.data_daily, data_tq, 
                                left_index=True, right_index=True)
    
        return self.data_daily
    
    def z_ratio_test(self, alpha=0.05):
        """
        test de Z-ratio pour detecter les jumps
        
        Parameters:
        alpha (float) : significance level
        
        Returns:
        pd.DataFrame: df contenant stats Z et variable binaire is_jump
        """

        # pi term dans denominateur du Z-ratio
        pi_term = (np.pi**2 / 4) + np.pi - 5
        
        # stats Z
        delta = 1/78  # à modifier -----------------------------------------------------------------------------------------------
        numerator = delta**(-1/2) * (self.data_daily['rv'] - self.data_daily['bpv']) / self.data_daily['rv']
        denominator = np.sqrt(pi_term * np.maximum(1, self.data_daily['tq'] / (self.data_daily['bpv']**2)))
        self.data_daily['z_ratio'] = numerator / denominator
        
        # est ce que les sauts sont significatif au niveau alpha
        critical_value = norm.ppf(1 - alpha/2)  # test bilateral
        self.data_daily['is_jump'] = np.abs(self.data_daily['z_ratio']) > critical_value
        
        return self.data_daily
    
    def continuous_sample_path(self):
        """
        calcule continuous sample path (CRV)
        
        Returns:
        pd.DataFrame: df contenant la composante continue de la volatilité (crv)
        """
        
        # Calculer CRV selon l'équation (5)
        # si is_jump est True prendre BPV sinon prendre RV
        crv = np.where(self.data_daily['is_jump'], 
                    self.data_daily['bpv'], 
                    self.data_daily['rv'])
        
        data_crv = pd.DataFrame(crv, index=self.data_daily.index, columns=['crv'])
        # ajoute crv au data_daily
        self.data_daily = pd.merge(self.data_daily, data_crv, 
                                left_index=True, right_index=True)
        
        return self.data_daily
    
    def jump_component(self):
        """
        calcule jump component (CJ)
        
        Returns:
        pd.DataFrame: df contenant la composante de jump dans la volatilité (cj)
        """
        
        # si is_jump est True, prendre max(0, RV-BPV) sinon 0
        cj = np.where(self.data_daily['is_jump'], 
                    np.maximum(0, self.data_daily['rv'] - self.data_daily['bpv']), 
                    0)
        
        data_cj = pd.DataFrame(cj, index=self.data_daily.index, columns=['cj'])
        
        # ajout cj au data_daily
        self.data_daily = pd.merge(self.data_daily, data_cj, 
                                left_index=True, right_index=True)
    
        return self.data_daily
    
    
    def all_volatility_measures(self, price_col='Ouv.', alpha=0.05):
        """
        calule toutes les mesures de volatilité et retourne un df
        
        Parameters:
        alpha (float) : significance level
        
        Returns:
        pd.DataFrame: df contenant toutes les mesures de volatilité
        """

        # return
        self.calculate_returns(price_col)
        # rv
        self.realized_volatility()
        # semi-variance
        self.semi_variance()
        # bpv
        self.bipower_variation()
        # tq
        self.realized_tripower_quarticity()
        # Z-ratio test
        self.z_ratio_test(alpha)
        # crv
        self.continuous_sample_path()
        # cj
        self.jump_component()
    
        return self.data_daily

if __name__ == '__main__':
    # test

    df_oil = pd.read_csv('data/raw/E-mini Crude Oil Futures_2007_2025.csv', delimiter=',', decimal=',')
    df_oil['Date'] = pd.to_datetime(df_oil['Date'], format='%d/%m/%Y')
    df_oil.index = df_oil['Date']
    df_oil.drop(columns=['Date'], inplace = True)

    rv_measure = RealizedVolatilityMeasure(df_oil)

    # return
    rv_measure.calculate_returns(price_col='Ouv.')
    # rv
    rv_measure.realized_volatility()
    # semi-variance
    rv_measure.semi_variance()
    # bpv
    rv_measure.bipower_variation()
    # tq
    rv_measure.realized_tripower_quarticity()
    # Z-ratio test
    rv_measure.z_ratio_test()
    # crv
    rv_measure.continuous_sample_path()
    # cj
    rv_measure.jump_component()