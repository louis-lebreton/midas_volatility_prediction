import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import minimize

class MIDASModel:
    """Classe qui implémente le modèle MIDAS pour prévision de volatilité"""
    
    def __init__(self, kmax=66, options_dict = {'maxiter': 1000}):
        """
        Parameters:
        kmax (int): nombre max de lags à prendre en compte
        options_dict (dict): options de l'optimisation
        """
        self.kmax = kmax
        self.options_dict = options_dict
        self.params_dict = None # dict des paramètres estimés des modeles : beta, gamma, theta
        
    def weighting_function(self, k, theta1, theta2):
        """
        fonction de pondération selon l'equation (8)
        f(k/kmax, theta1, theta2)
        
        Parameters:
        k (int): indice de retard
        theta1, theta2 (float): params de la weighting function
        """
        if k == self.kmax:
            z = 1 - 1e-10  # pour éviter z = 1
        else:
            z = k/self.kmax
        numerator = z**(theta1-1) * (1-z)**(theta2-1)
        denominator = gamma(theta1) * gamma(theta2) / gamma(theta1 + theta2)
        return numerator / denominator
    
    def normalize_weights(self, theta_params):
        """
        b(k,θRV) : normalise les poids pour qu'ils somment à 1 selon l'equation (8)
        
        Parameters:
        theta_params (tuple): params de la weighting function
        """
        theta1, theta2 = theta_params
        weights = np.array([self.weighting_function(k, theta1, theta2) for k in range(1, self.kmax+1)])

        return weights / np.sum(weights)
    
    def weighted_sum(self, X, theta_params):
        """
        calcule la somme pondérée des X selon les poids déterminés par theta_params
        
        Parameters:
        X (array): série de data à pondérer (ici : RV)
        theta_params (tuple): params de la weighting function

        Returns :
        float: produit scalaire entre poids et vecteur
        """
        weights = self.normalize_weights(theta_params) # obtention des poids pour chaque k
        if len(X) < self.kmax:
            padded_X = np.pad(X, (0, self.kmax - len(X)), 'constant')
        else:
            padded_X = X[-self.kmax:]
        return np.dot(weights, padded_X)
    
    # Model 0 : MIDAS-RV
    def fit_midas_rv(self, y, X, horizon=1):
        """
        ajuste le modèle MIDAS-RV (Model 0) selon l'équation (7)
        
        Parameters:
        y (array): série à prédire (RV futur)
        X (array): série RV passés
        horizon (int): horizon de prévision (1, 5 ou 22 jours)
        
        Returns:
        dict: params estimés (beta0, beta1, theta1, theta2)
        """
        def loss_function(params_dict):
            """
            nested function pour calculer la loss
            """
            # 4 paramètres ici
            beta0, beta1, theta1, theta2 = params_dict
            
            predictions = []
            
            for t in range(self.kmax, len(X) - horizon):
                X_t = X[t-self.kmax:t] # X_t qui prend en compte tous les lags
                weighted_X = self.weighted_sum(X_t, (theta1, theta2))
                pred = beta0 + beta1 * weighted_X
                predictions.append(pred)
            
            # valeurs réelles
            actual = y[:len(X) - horizon - self.kmax]
            
            # calcul de la loss: HMSE
            actual = np.where(actual == 0, 1e-8, actual)
            hmse = np.mean((1 - np.array(predictions) / actual) ** 2)
            return hmse
        
        # choix des paramètres initiaux
        initial_params = [0.1, 0.5, 1.0, 1.0]  # beta0, beta1, theta1, theta2
        # contraintes pour theta1 et theta2 (doivent être > 0 )
        bounds = [(None, None), (None, None), (0.01, None), (0.01, None)]
        # optimisation
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options=self.options_dict)
        # best params
        beta0, beta1, theta1, theta2 = result.x

        # dict de result
        self.params_dict = {
            'beta0': beta0,
            'beta1': beta1,
            'theta_RV': (theta1, theta2)
        }
        
        return self.params_dict
    
    # Model 1 : MIDAS-RS
    def fit_midas_rs(self, y, X_pos, X_neg, horizon=1):
        """
        ajuste le modèle MIDAS-RS (modèle 1) selon l'équation (9)
        
        Parameters:
        y (array): série à prédire (RV futur)
        X_pos (array): série des semi-variances positives passées
        X_neg (array): série des semi-variances négatives passées
        horizon (int): horizon de prévision (1, 5 ou 22 jours)
        
        Returns:
        dict: params estimés
        """
        def loss_function(params):
            """
            nested function pour calculer la loss
            """
            beta0, beta1, beta2, theta1_pos, theta2_pos, theta1_neg, theta2_neg = params
            
            predictions = []
            
            for t in range(self.kmax, min(len(X_pos), len(X_neg)) - horizon):
                X_pos_t = X_pos[t-self.kmax:t]
                X_neg_t = X_neg[t-self.kmax:t]
                weighted_X_pos = self.weighted_sum(X_pos_t, (theta1_pos, theta2_pos))
                weighted_X_neg = self.weighted_sum(X_neg_t, (theta1_neg, theta2_neg))
                pred = beta0 + beta1 * weighted_X_pos + beta2 * weighted_X_neg
                predictions.append(pred)
            
            # valeurs réelles
            actual = y[:min(len(X_pos), len(X_neg)) - horizon - self.kmax]
            # calcul de la loss: HMSE
            actual = np.where(actual == 0, 1e-8, actual)
            hmse = np.mean((1 - np.array(predictions) / actual) ** 2)
            return hmse
        
        # choix des paramètres initiaux
        initial_params = [0.1, 0.5, 0.5, 1.7, 1.1, 1.2, 1.3]
        
        # contraintes les theta doivent être > 0
        bounds = [(None, None), (None, None), (None, None), 
                  (0.01, None), (0.01, None), (0.01, None), (0.01, None)]
        
        # optimisation
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options=self.options_dict)
        # best params
        beta0, beta1, beta2, theta1_pos, theta2_pos, theta1_neg, theta2_neg = result.x

        # dict de result
        self.params_dict = {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'theta_RV_pos': (theta1_pos, theta2_pos),
            'theta_RV_neg': (theta1_neg, theta2_neg)
        }
        
        return self.params_dict
    
    # Model 2 : MIDAS-CJ
    def fit_midas_cj(self, y, X_crv, X_cj, horizon=1):
        """
        ajuste le modèle MIDAS-CJ (modèle 2) selon l'équation (10)
        
        Parameters:
        y (array): RV future à prédire
        X_crv (array): série des composantes continues passées
        X_cj (array): série des composantes de saut passées
        horizon (int): horizon de prévision (1, 5 ou 22 jours)
        
        Returns:
        dict: paramètres estimés
        """
        def loss_function(params):
            """
            nested function pour calculer la loss
            """
            beta0, beta1, beta2, theta1_crv, theta2_crv, theta1_cj, theta2_cj = params
            
            predictions = []
            
            for t in range(self.kmax, min(len(X_crv), len(X_cj)) - horizon):
                X_crv_t = X_crv[t-self.kmax:t]
                X_cj_t = X_cj[t-self.kmax:t]
                
                weighted_X_crv = self.weighted_sum(X_crv_t, (theta1_crv, theta2_crv))
                weighted_X_cj = self.weighted_sum(X_cj_t, (theta1_cj, theta2_cj))
                
                pred = beta0 + beta1 * weighted_X_crv + beta2 * weighted_X_cj
                predictions.append(pred)
            
            # valeurs réelles
            actual = y[:min(len(X_crv), len(X_cj)) - horizon - self.kmax]
            # calcul de la loss: HMSE
            actual = np.where(actual == 0, 1e-8, actual)
            hmse = np.mean((1 - np.array(predictions) / actual) ** 2)
            return hmse
        
        # paramètres initiaux
        initial_params = [0.1, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
        # contraintes : theta > 0
        bounds = [(None, None), (None, None), (None, None), 
                  (0.01, None), (0.01, None), (0.01, None), (0.01, None)]
        # optimisation
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options=self.options_dict)
        # best params
        beta0, beta1, beta2, theta1_crv, theta2_crv, theta1_cj, theta2_cj = result.x
        
        self.params_dict = {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'theta_CRV': (theta1_crv, theta2_crv),
            'theta_CJ': (theta1_cj, theta2_cj)
        }
        
        return self.params_dict
    
    # Model 3 : MIDAS-RV-GPR 
    def fit_midas_rv_gpr(self, y, X_rv, X_gpr, horizon=1):
        """
        ajuste le modèle MIDAS-RV-GPR (modèle 3) selon l'équation (11)
        
        Parameters:
        y (array): RV futur à prédire
        X_rv (array): série des volatilités réalisées passées
        X_gpr (array): série de l'indice GPR
        horizon (int): horizon de prévision (1, 5 ou 22 jours)
        
        Returns:
        dict: params estimés
        """
        def loss_function(params):
            """
            nested function pour calculer la loss
            """
            beta0, beta1, gamma1, theta1_rv, theta2_rv, theta1_gpr, theta2_gpr = params
            
            predictions = []
            
            for t in range(self.kmax, min(len(X_rv), len(X_gpr)) - horizon):
                X_rv_t = X_rv[t-self.kmax:t]
                X_gpr_t = X_gpr[t-self.kmax:t]
                
                weighted_X_rv = self.weighted_sum(X_rv_t, (theta1_rv, theta2_rv))
                weighted_X_gpr = self.weighted_sum(X_gpr_t, (theta1_gpr, theta2_gpr))
                
                pred = beta0 + beta1 * weighted_X_rv + gamma1 * weighted_X_gpr
                predictions.append(pred)
            # valeurs réalisés
            actual = y[:min(len(X_rv), len(X_gpr)) - horizon - self.kmax]
            # calcul de la loss: HMSE
            actual = np.where(actual == 0, 1e-8, actual)
            hmse = np.mean((1 - np.array(predictions) / actual) ** 2)
            return hmse
        
        # estimation des paramètres initiaux
        initial_params = [0.1, 0.5, 0.1, 1.0, 1.0, 1.0, 1.0]
        # contraintes : theta > 0
        bounds = [(None, None), (None, None), (None, None), 
                  (0.01, None), (0.01, None), (0.01, None), (0.01, None)]
        
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options=self.options_dict)
        # best params
        beta0, beta1, gamma1, theta1_rv, theta2_rv, theta1_gpr, theta2_gpr = result.x
        
        self.params_dict = {
            'beta0': beta0,
            'beta1': beta1,
            'gamma1': gamma1,
            'theta_RV': (theta1_rv, theta2_rv),
            'theta_GPR': (theta1_gpr, theta2_gpr)
        }
        
        return self.params_dict
    
    # Model 4 : MIDAS-RS-GPR
    def fit_midas_rs_gpr(self, y, X_pos, X_neg, X_gpr, horizon=1):
        """
        ajuste le modèle MIDAS-RS-GPR (model 4) selon l'équation (12)
        
        Parameters:
        y (array): RV futur à prédire
        X_pos (array): série des semi-variances positives
        X_neg (array): série des semi-variances négatives
        X_gpr (array): série de l'indice GPR
        horizon (int): horizon de prévision (1, 5 ou 22 jours)
        
        Returns:
        dict: params estimés
        """
        def loss_function(params):
            """
            nested function pour calculer la loss
            """
            beta0, beta1, beta2, gamma1 = params[:4]
            theta1_pos, theta2_pos, theta1_neg, theta2_neg, theta1_gpr, theta2_gpr = params[4:]
            
            predictions = []
            
            min_len = min(len(X_pos), len(X_neg), len(X_gpr))
            for t in range(self.kmax, min_len - horizon):
                X_pos_t = X_pos[t-self.kmax:t]
                X_neg_t = X_neg[t-self.kmax:t]
                X_gpr_t = X_gpr[t-self.kmax:t]
                
                weighted_X_pos = self.weighted_sum(X_pos_t, (theta1_pos, theta2_pos))
                weighted_X_neg = self.weighted_sum(X_neg_t, (theta1_neg, theta2_neg))
                weighted_X_gpr = self.weighted_sum(X_gpr_t, (theta1_gpr, theta2_gpr))
                
                pred = beta0 + beta1 * weighted_X_pos + beta2 * weighted_X_neg + gamma1 * weighted_X_gpr
                predictions.append(pred)
            
            # valeurs realisées
            actual = y[:min_len - horizon - self.kmax]
            # calcul de la loss: HMSE
            actual = np.where(actual == 0, 1e-8, actual)
            hmse = np.mean((1 - np.array(predictions) / actual) ** 2)
            return hmse
        
        # estimation des paramètres initiaux
        initial_params = [0.1, 0.5, 0.5, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # contraintes : theta > 0
        bounds = [(None, None), (None, None), (None, None), (None, None),
                  (0.01, None), (0.01, None), (0.01, None), (0.01, None), (0.01, None), (0.01, None)]
        # optimization
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options=self.options_dict)
        # best params
        params_dict = result.x
        beta0, beta1, beta2, gamma1 = params_dict[:4]
        theta1_pos, theta2_pos, theta1_neg, theta2_neg, theta1_gpr, theta2_gpr = params_dict[4:]
        
        self.params_dict = {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'gamma1': gamma1,
            'theta_RV_pos': (theta1_pos, theta2_pos),
            'theta_RV_neg': (theta1_neg, theta2_neg),
            'theta_GPR': (theta1_gpr, theta2_gpr)
        }
        
        return self.params_dict
    
    # Model 5 : MIDAS-CJ-GPR
    def fit_midas_cj_gpr(self, y, X_crv, X_cj, X_gpr, horizon=1):
        """
        ajuste le modèle MIDAS-CJ-GPR (modèle 5) selon l'équation (13)
        
        Parameters:
        y (array): RV futur à prédire
        X_crv (array): série des composantes continues
        X_cj (array): série des composantes de saut
        X_gpr (array): série de l'indice GPR
        horizon (int): horizon de prévision (1, 5 ou 22 jours)
        
        Returns:
        dict: params estimés
        """
        def loss_function(params):
            """
            nested function pour calculer la loss
            """
            beta0, beta1, beta2, gamma1 = params[:4]
            theta1_crv, theta2_crv, theta1_cj, theta2_cj, theta1_gpr, theta2_gpr = params[4:]
            
            predictions = []
            
            min_len = min(len(X_crv), len(X_cj), len(X_gpr))
            for t in range(self.kmax, min_len - horizon):
                X_crv_t = X_crv[t-self.kmax:t]
                X_cj_t = X_cj[t-self.kmax:t]
                X_gpr_t = X_gpr[t-self.kmax:t]
                
                weighted_X_crv = self.weighted_sum(X_crv_t, (theta1_crv, theta2_crv))
                weighted_X_cj = self.weighted_sum(X_cj_t, (theta1_cj, theta2_cj))
                weighted_X_gpr = self.weighted_sum(X_gpr_t, (theta1_gpr, theta2_gpr))
                
                pred = beta0 + beta1 * weighted_X_crv + beta2 * weighted_X_cj + gamma1 * weighted_X_gpr
                predictions.append(pred)

            # valeurs réalisées
            actual = y[:min_len - horizon - self.kmax]
            # calcul de la loss: HMSE
            actual = np.where(actual == 0, 1e-8, actual)
            hmse = np.mean((1 - np.array(predictions) / actual) ** 2)
            return hmse
        
        # estimation des paramètres initiaux
        initial_params = [0.1, 0.5, 0.5, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # contraintes : theta > 0
        bounds = [(None, None), (None, None), (None, None), (None, None),
                  (0.01, None), (0.01, None), (0.01, None), (0.01, None), (0.01, None), (0.01, None)]
        # optimisation
        result = minimize(loss_function, initial_params, bounds=bounds, method='L-BFGS-B', options=self.options_dict)
        # best params
        params_dict = result.x
        beta0, beta1, beta2, gamma1 = params_dict[:4]
        theta1_crv, theta2_crv, theta1_cj, theta2_cj, theta1_gpr, theta2_gpr = params_dict[4:]
        
        self.params_dict = {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'gamma1': gamma1,
            'theta_CRV': (theta1_crv, theta2_crv),
            'theta_CJ': (theta1_cj, theta2_cj),
            'theta_GPR': (theta1_gpr, theta2_gpr)
        }
        
        return self.params_dict
    
    
    def predict(self, X:pd.DataFrame, model_type='MIDAS-RV', start_idx=None, gpr_str = 'gpr'):
        """
        réalise la prédiction à partir des paramètres estimés
        attention : choisir le bonne fonction de fit associé à un modèle avant

        Parameters:
        X (df): df contenant les séries temporelles nécessaires selon le modèle
        model_type (str): choix du modèle à utiliser pour la prédiction
        start_idx (int, optional): indice de départ pour prédiction (default = kmax)
        gpr_str (str) : choix de l'indice GPR utilisé
        
        Returns:
        array: Prédictions de volatilité pour chaque point temporel
        """
        # pas de start_idx => commence à partir de kmax
        if start_idx is None:
            start_idx = self.kmax
        
        predictions = []
        
        if model_type == 'MIDAS-RV':
            rv = X['rv'].values
            for t in range(start_idx, len(rv)):
                X_t = rv[t-self.kmax:t]
                weighted_rv = self.weighted_sum(X_t, self.params_dict['theta_RV'])
                pred = self.params_dict['beta0'] + self.params_dict['beta1'] * weighted_rv
                predictions.append(pred)
        
        elif model_type == 'MIDAS-RS':
            rv_pos = X['rv_pos'].values
            rv_neg = X['rv_neg'].values
            for t in range(start_idx, min(len(rv_pos), len(rv_neg))):
                X_pos_t = rv_pos[t-self.kmax:t]
                X_neg_t = rv_neg[t-self.kmax:t]
                weighted_rv_pos = self.weighted_sum(X_pos_t, self.params_dict['theta_RV_pos'])
                weighted_rv_neg = self.weighted_sum(X_neg_t, self.params_dict['theta_RV_neg'])
                pred = self.params_dict['beta0'] + self.params_dict['beta1'] * weighted_rv_pos + self.params_dict['beta2'] * weighted_rv_neg
                predictions.append(pred)
        
        elif model_type == 'MIDAS-CJ':
            crv = X['crv'].values
            cj = X['cj'].values
            for t in range(start_idx, min(len(crv), len(cj))):
                X_crv_t = crv[t-self.kmax:t]
                X_cj_t = cj[t-self.kmax:t]
                weighted_crv = self.weighted_sum(X_crv_t, self.params_dict['theta_CRV'])
                weighted_cj = self.weighted_sum(X_cj_t, self.params_dict['theta_CJ'])
                pred = self.params_dict['beta0'] + self.params_dict['beta1'] * weighted_crv + self.params_dict['beta2'] * weighted_cj
                predictions.append(pred)
        
        elif model_type == 'MIDAS-RV-GPR':
            rv = X['rv'].values
            gpr = X[gpr_str].values
            for t in range(start_idx, min(len(rv), len(gpr))):
                X_rv_t = rv[t-self.kmax:t]
                X_gpr_t = gpr[t-self.kmax:t]
                weighted_rv = self.weighted_sum(X_rv_t, self.params_dict['theta_RV'])
                weighted_gpr = self.weighted_sum(X_gpr_t, self.params_dict['theta_GPR'])
                pred = self.params_dict['beta0'] + self.params_dict['beta1'] * weighted_rv + self.params_dict['gamma1'] * weighted_gpr
                predictions.append(pred)
        
        elif model_type == 'MIDAS-RS-GPR':
            rv_pos = X['rv_pos'].values
            rv_neg = X['rv_neg'].values
            gpr = X[gpr_str].values
            for t in range(start_idx, min(len(rv_pos), len(rv_neg), len(gpr))):
                X_pos_t = rv_pos[t-self.kmax:t]
                X_neg_t = rv_neg[t-self.kmax:t]
                X_gpr_t = gpr[t-self.kmax:t]
                weighted_rv_pos = self.weighted_sum(X_pos_t, self.params_dict['theta_RV_pos'])
                weighted_rv_neg = self.weighted_sum(X_neg_t, self.params_dict['theta_RV_neg'])
                weighted_gpr = self.weighted_sum(X_gpr_t, self.params_dict['theta_GPR'])
                pred = self.params_dict['beta0'] + self.params_dict['beta1'] * weighted_rv_pos + self.params_dict['beta2'] * weighted_rv_neg + self.params_dict['gamma1'] * weighted_gpr
                predictions.append(pred)
        
        elif model_type == 'MIDAS-CJ-GPR':
            crv = X['crv'].values
            cj = X['cj'].values
            gpr = X[gpr_str].values
            for t in range(start_idx, min(len(crv), len(cj), len(gpr))):
                X_crv_t = crv[t-self.kmax:t]
                X_cj_t = cj[t-self.kmax:t]
                X_gpr_t = gpr[t-self.kmax:t]
                weighted_crv = self.weighted_sum(X_crv_t, self.params_dict['theta_CRV'])
                weighted_cj = self.weighted_sum(X_cj_t, self.params_dict['theta_CJ'])
                weighted_gpr = self.weighted_sum(X_gpr_t, self.params_dict['theta_GPR'])
                pred = self.params_dict['beta0'] + self.params_dict['beta1'] * weighted_crv + self.params_dict['beta2'] * weighted_cj + self.params_dict['gamma1'] * weighted_gpr
                predictions.append(pred)
        
        else:
            raise ValueError(f"Type de modèle non reconnu: {model_type}")
        
        return np.array(predictions)
        
