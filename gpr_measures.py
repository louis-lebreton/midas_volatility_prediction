import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class GPRDecomposer:
    """
    Cette classe permet de décomposer la série GPR index en 2 composantes :
    Expected GPR (EG) : the estimated GPRt as the expected geopolitical risk
    Shocked GPR (SG) : the residuals (εt) as the shocked geopolitical risk
    en utilisant un modèle AR(1) : GPRt = v1 + v2*GPRt-1 + εt (début de la page 6 de l'article)
    """
    
    def __init__(self):
        """Initialisation de la classe."""
        self.model = None
        self.results = None
        self.v1 = None  # constante (v1)
        self.v2 = None  # coeff d'autocorrélation (v2)
    
    def fit(self, gpr_series):
        """
        Ajuster le modèle AR(1) sur la série GPR.
        
        Args:
            gpr_series (pd.Series): serie temporelle de l'indice GPR
        """
        # modele d'autoregression à 1 lag
        self.model = AutoReg(gpr_series, lags=1)
        self.results = self.model.fit()
        
        # params du modèle estimées
        self.v1 = self.results.params[0]
        self.v2 = self.results.params[1]
        
        return self
    
    def transform(self, gpr_series):
        """
        décompose la série index GPR en expected et shocked
        
        Args:
            gpr_series (pd.Series): serie temporelle de l'indice GPR
            
        Returns:
            pd.DataFrame: df avec en plus, expected et shocked
        """
        # Prédictions du modèle AR(1) (expected GPR)
        expected_gpr = self.results.predict(start=1, end=len(gpr_series))
        expected_gpr = pd.Series(expected_gpr, index=gpr_series.index[1:])
        
        # Résidus du modèle AR(1) (shocked GPR)
        shocked_gpr = self.results.resid
        shocked_gpr = pd.Series(shocked_gpr, index=gpr_series.index[1:])
        
        # Créer le DataFrame résultat
        result_df = pd.DataFrame({
            'gpr': gpr_series[1:],
            'gpr_expected': expected_gpr,
            'gpr_shocked': shocked_gpr
        })
        
        return result_df
    
    def fit_transform(self, gpr_series):
        """
        wrapper qui train le modèle AR et transformer l'indice GPR en une seule étape
        
        Args:
            gpr_series (pd.Series): serie temporelle de l'indice GPR
            
        Returns:
            pd.DataFrame: df avec en plus, expected et shocked
        """
        return self.fit(gpr_series).transform(gpr_series)
    

if __name__ == '__main__':
    df_gpr = pd.read_excel('data/raw/data_gpr_daily_recent.xls', usecols=range(9))
    df_gpr['DAY'] = pd.to_datetime(df_gpr['DAY'], format='%Y%m%d')
    df_gpr.index = df_gpr['DAY']
    df_gpr.drop(columns=['DAY'], inplace = True)

    decomposer = GPRDecomposer()
    df_gpr_augmente = decomposer.fit_transform(df_gpr['GPRD'])
    df_gpr_augmente.head()
