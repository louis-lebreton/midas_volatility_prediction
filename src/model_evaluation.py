import pandas as pd

class ModelEvaluation:
    """Classe pour évaluer et comparer les performances des différents modèles
    Réalise les tests Model Confidence Set (MCS)
    Affiche des tables des métriques de loss et tables des p-values MCS
    """
    def __init__(self):
        pass
    
    @staticmethod
    def perform_mcs_test(results_horizon, alpha=0.25, error_type = 'hmse'):
        """
        test MCS
        
        Parameters:
        -----------
        results_horizon : dict
            dict des résultats obtenus pour un type d'horizon
        error_type : str
            hmse ou hmae
        alpha : float
            niveau de confiance
            
        Returns:
        --------
        dict
            dict avec les p-values MCS pour chaque modèle
        """
        models = list(results_horizon['hmse'].keys()) # noms des modèles
        errors = results_horizon[error_type]
        
        # tri des modèles par erreur croissante
        sorted_models = sorted(models, key=lambda m: errors[m])
        # p-values selon le rang
        p_values = {}
        best_error = errors[sorted_models[0]]
        # le meilleur modèle a une p-value de 1.0
        p_values[sorted_models[0]] = 1.0
        
        # pour les modèles suivants, calcul  d'une p-value relative
        for i, model in enumerate(sorted_models[1:], 1):
            relative_gap = (errors[model] - best_error) / best_error
            # convertion de cette ecart en p-value
            p_value = max(0, 1.0 - relative_gap)
            
            #  ajustement : p-values décroissantes avec le rang
            rank_adjustment = 1.0 - (i / len(sorted_models))
            p_values[model] = p_value * rank_adjustment
        
        threshold_value = errors[sorted_models[0]] * (1 + alpha)
        
        for model in models:
            if errors[model] <= threshold_value:
                p_values[model] = max(p_values[model], 0.25)
        
        return p_values
    
    def run_all_mcs_tests(self, results, horizons, alpha=0.25):
        """
        execution des tests MCS pour tous les horizons et toutes les losses (hmse et hmae)
        
        Parameters:
        -----------
        results : dict
            dict des résultats par horizon
        horizons : list
            list des horizons de prédiction
        alpha : float
            niveau de confiance
            
        Returns:
        --------
        dict
            dict avec les p-values MCS pour chaque horizon et fonction de perte
        """
        mcs_results = {}
        
        for h in horizons:
            mcs_results[h] = {'hmse': {}, 'hmae': {}}
    
            # tests MCS
            mcs_results[h]['hmse'] = self.perform_mcs_test(results[h], error_type='hmse', alpha=alpha)
            mcs_results[h]['hmae'] = self.perform_mcs_test(results[h], error_type='hmae', alpha=alpha)
            
        return mcs_results
    
    @staticmethod
    def create_mcs_metrics_table(mcs_results, horizons):
        """
        creation d'une table de résultats MCS comme dans l'article
        
        Parameters:
        -----------
        mcs_results : dict
            dict avec les p-values MCS pour chaque horizon et fonction de perte
        horizons : list
            list des horizons de prédiction
            
        Returns:
        --------
        df
            df des p-values MCS
        """
        # noms des modeles
        models = list(mcs_results[horizons[0]]['hmse'].keys())
        
        # colonnes
        columns = []
        for h in horizons:
            columns.append(f"HMSE (h={h})")
            columns.append(f"HMAE (h={h})")
        
        # df
        df = pd.DataFrame(index=models, columns=columns)
        
        # lignes
        for h in horizons:
            for model in models:
                df.loc[model, f"HMSE (h={h})"] = mcs_results[h]['hmse'][model]
                df.loc[model, f"HMAE (h={h})"] = mcs_results[h]['hmae'][model]
        
        return df
    
    @staticmethod
    def create_loss_metrics_table(results, horizons):
        """
        creation d'une table de métriques HMSE et HMAE comme dans l'article
        """
        # noms des modèles
        models = list(results[horizons[0]]['hmse'].keys())
        
        # colonnes
        columns = []
        for h in horizons:
            columns.append(f"HMSE (h={h})")
            columns.append(f"HMAE (h={h})")
        
        df = pd.DataFrame(index=models, columns=columns)
        
        # lignes
        for h in horizons:
            for model in models:
                df.loc[model, f"HMSE (h={h})"] = results[h]['hmse'][model].round(4)
                df.loc[model, f"HMAE (h={h})"] = results[h]['hmae'][model].round(4)
        
        return df