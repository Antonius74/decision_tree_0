Ecco il file README.md richiesto:

```markdown
# Decision Tree - Spiegazione Didattica

## Introduzione e Teoria

Un **Decision Tree** (Albero Decisionale) è un algoritmo di machine learning che crea un modello di predizione strutturato come un albero. Immagina di dover classificare se un frutto è una mela o un'arancia basandoti sulle sue caratteristiche:

- Se il colore è rosso → probabilmente è una mela
- Se il colore è arancione → probabilmente un'arancia
- Se il colore è verde → controlla la dimensione:
  - Se grande → potrebbe essere una mela verde
  - Se piccolo → potrebbe essere un'arancia non matura

L'albero decisionale formalizza questo processo decisionale attraverso una struttura ad albero dove:

- **Nodi interni**: rappresentano test sulle caratteristiche (es: "colore = rosso?")
- **Rami**: rappresentano l'esito del test (es: "sì" o "no")
- **Foglie**: rappresentano la decisione finale (es: "mela")

Matematicamente, l'algoritmo cerca di massimizzare la **purezza** dei nodi utilizzando metriche come l'**impurità di Gini**:

$$
Gini = 1 - \sum_{i=1}^{c} (p_i)^2
$$

dove $p_i$ è la proporzione degli elementi della classe $i$ nel nodo. Un Gini=0 indica un nodo perfettamente puro (tutti gli elementi della stessa classe).

## Dati Utilizzati (Input/Output)

### Dataset Breast Cancer Wisconsin
Lo script utilizza un dataset reale di diagnostica medica che contiene:

- **569 campioni** di tessuto mammario
- **30 caratteristiche** numeriche per ogni campione 
- **2 classi di output**: "maligno" (0) e "benigno" (1)

### Esempio di Dato Grezzo
Un singolo campione potrebbe essere rappresentato così:

```python
# Caratteristiche (input)
[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 
 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 
 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 
 0.2654, 0.4601, 0.1189]

# Classe target (output): 0 → maligno
```

Le caratteristiche rappresentano misurazioni quantitative come:
- **Raggio medio**: 17.99 (dimensione media del nucleo cellulare)
- **Textura media**: 10.38 (variazione del colore)
- **Perimetro medio**: 122.8
- **Area media**: 1001.0

## Analisi del Codice

### 1. Preprocessing dei Dati

```python
def preprocessamento(self, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=self.random_state, stratify=y
    )
    
    self.X_train = self.scaler.fit_transform(X_train)
    self.X_test = self.scaler.transform(X_test)
```

**Divisione Train/Test**: 
- 70% dati per il training (398 campioni)
- 30% dati per il test (171 campioni)
- `stratify=y` mantiene la proporzione originale delle classi

**Standardizzazione**: 
- Trasforma tutte le features sulla stessa scala
- Evita che features con valori numerici più grandi dominino il processo decisionale

$$
x_{standard} = \frac{x - \mu}{\sigma}
$$

### 2. Training del Modello

```python
def addestra_modello(self, max_depth=5, min_samples_split=10):
    self.model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=self.random_state
    )
    self.model.fit(self.X_train, self.y_train)
```

**Parametri di Controllo**:
- `max_depth=5`: limita la profondità massima dell'albero per evitare overfitting
- `min_samples_split=10`: richiede almeno 10 campioni per dividere un nodo

**Processo di Costruzione**:
1. Inizia con tutti i campioni nel nodo radice
2. Per ogni feature, calcola la riduzione dell'impurità
3. Scegli la feature che massimizza la riduzione
4. Dividi il nodo in base al valore di soglia ottimale
5. Ripeti ricorsivamente fino a raggiungere i criteri di stop

### 3. Valutazione delle Performance

```python
def valuta_modello(self):
    y_pred = self.model.predict(self.X_test)
    accuracy_test = accuracy_score(self.y_test, y_pred)
```

**Metriche Principali**:
- **Accuracy**: percentuale di classificazioni corrette
- **Matrice di Confusione**: confronta predizioni vs valori reali
- **Classification Report**: precision, recall, F1-score per ogni classe

**Esempio di Matrice di Confusione**:
```
[[ 60   3]   → 60 maligni corretti, 3 falsi benigni
 [  2 106]]  → 106 benigni corretti, 2 falsi maligni
```

### 4. Analisi dell'Importanza delle Features

```python
def importanza_features(self, feature_names):
    importances = self.model.feature_importances_
```

L'algoritmo assegna un punteggio di importanza a ogni feature basato su:
- Quante volte è usata per dividere i nodi
- Quanta impurità riduce complessivamente

Le feature più importanti saranno quelle che meglio separano le classi "maligno" e "benigno".

## Esempio Pratico Semplificato

Immagina di classificare frutti basandoti su due caratteristiche:

| Colore | Dimensione | Tipo    |
|--------|------------|---------|
| Rosso  | Grande     | Mela    |
| Rosso  | Piccolo    | Ciliegia|
| Giallo | Grande     | Banana  |
| Verde  | Grande     | Mela    |

L'albero decisionale imparerebbe automaticamente regole come:
- Se colore = Rosso e dimensione = Grande → Mela
- Se colore = Rosso e dimensione = Piccola → Ciliegia
- Se colore = Giallo → Banana

Questo principio, applicato alle 30 caratteristiche mediche del dataset, permette di diagnosticare tumori con alta accuratezza.
```