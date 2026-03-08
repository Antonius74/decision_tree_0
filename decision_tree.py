Ecco lo script Python completo e modulare per implementare un algoritmo Decision Tree:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class DecisionTreePipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def carica_dati(self):
        """Carica il dataset Breast Cancer Wisconsin"""
        dataset = load_breast_cancer()
        X = dataset.data
        y = dataset.target
        feature_names = dataset.feature_names
        target_names = dataset.target_names
        
        print(f"Dimensionalità dataset: {X.shape}")
        print(f"Numero di features: {X.shape[1]}")
        print(f"Classi target: {target_names}")
        print(f"Distribuzione classi: {np.bincount(y)}")
        
        return X, y, feature_names, target_names
    
    def preprocessamento(self, X, y):
        """Divide i dati e applica preprocessing"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Dimensionalità training set: {self.X_train.shape}")
        print(f"Dimensionalità test set: {self.X_test.shape}")
    
    def addestra_modello(self, max_depth=None, min_samples_split=2):
        """Addestra il modello Decision Tree"""
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=self.random_state
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print("Modello Decision Tree addestrato con successo")
        print(f"Profondità albero: {self.model.get_depth()}")
        print(f"Numero foglie: {self.model.get_n_leaves()}")
    
    def valuta_modello(self):
        """Valuta le performance del modello"""
        if self.model is None:
            raise ValueError("Modello non addestrato")
        
        y_pred = self.model.predict(self.X_test)
        y_pred_train = self.model.predict(self.X_train)
        
        accuracy_test = accuracy_score(self.y_test, y_pred)
        accuracy_train = accuracy_score(self.y_train, y_pred_train)
        
        print("\n" + "="*50)
        print("VALIDAZIONE MODELLO DECISION TREE")
        print("="*50)
        print(f"Accuracy Training Set: {accuracy_train:.4f}")
        print(f"Accuracy Test Set: {accuracy_test:.4f}")
        
        print("\nMATRICE DI CONFUSIONE:")
        print(confusion_matrix(self.y_test, y_pred))
        
        print("\nREPORT DI CLASSIFICAZIONE:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=load_breast_cancer().target_names))
        
        return y_pred
    
    def visualizza_albero(self, feature_names, target_names):
        """Visualizza la struttura dell'albero decisionale"""
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                 feature_names=feature_names,
                 class_names=target_names,
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title("Struttura Albero Decisionale")
        plt.show()
    
    def importanza_features(self, feature_names):
        """Analizza l'importanza delle features"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTOP 10 FEATURES PER IMPORTANZA:")
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), importances[indices[:10]])
        plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
        plt.title("Top 10 Features per Importanza")
        plt.tight_layout()
        plt.show()

def main():
    """Funzione principale per eseguire la pipeline completa"""
    pipeline = DecisionTreePipeline(random_state=42)
    
    try:
        X, y, feature_names, target_names = pipeline.carica_dati()
        pipeline.preprocessamento(X, y)
        pipeline.addestra_modello(max_depth=5, min_samples_split=10)
        y_pred = pipeline.valuta_modello()
        pipeline.importanza_features(feature_names)
        pipeline.visualizza_albero(feature_names, target_names)
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    main()


Questo script implementa una pipeline completa per l'uso di Decision Tree con le seguenti caratteristiche:

**Caricamento dati**: Utilizza il dataset Breast Cancer Wisconsin, ideale per classificatione binaria.

**Preprocessing**: Divisione train/test (70/30) con stratificazione e scaling delle features.

**Modellazione**: DecisionTreeClassifier con parametri configurabili per evitare overfitting.

**Valutazione**: Metriche complete includendo accuracy, matrice di confusione e classification report.

**Visualizzazione**: Plot dell'albero decisionale e analisi importanza features.

**Modularità**: Classe ben strutturata che permette facile riutilizzo e modifica dei parametri.