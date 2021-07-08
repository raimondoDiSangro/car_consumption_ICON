# Car consumption ICON project

### Descrizione


### Dataset
- I dati a disposizione contengono specifiche tecniche di una serie di macchine.
 - Il dataset è stato scaricato dal Repository UCI Machine learning. https://archive.ics.uci.edu/ml/datasets/auto+mpg
- Contiene le specifiche tecniche di auto dal 1970 al 1982.
- Considerata l'età dei dati, ho ritenuto opportuno aggiornare il dataset personalmente con dati di vetture moderne.


### Contenuto dataset originale
  1. ```mpg: consumo in miglia per gallone (miglio = 1.609 km circa, gallone = 3,785 litri)```
  2. ```cylinders: numero di cilindri ```
  3. ```displacement: cilindrata in cu in (pollici cubo, 1 cu in = 16,387cc circa)```
  4. ```horsepower: potenza in cavalli```
  5.```weight: peso in libbre (1lb = 0,453 kg circa)```
  6. ```acceleration: 0-60mph espressa in secondi```
  7. ```model year: anno di produzione(dal 1970 al 1982)```
  8. ```origin: area di origine (1- USA, 2-Europa, 3-Asia)```
  9. ```car name: stringa contenente il nome dell'auto```


### Contenuto dataset aggiornato
  1. ```kpl: consumo in chilometri per litro```
  2. ```cylinders: numero di cilindri ```
  3. ```displacement: cilindrata in cc```
  4. ```horsepower: potenza in cavalli```
  5.```weight: peso in kg)```
  6. ```acceleration: 0-100kph espressa in secondi```
  7. ```model year: anno di produzione(dal 1970 al 2021)```
  8. ```origin: area di origine (1- USA, 2-Europa, 3-Asia)```
  9. ```car name: stringa contenente il nome dell'auto```



### Requisiti
- librerie ```numpy```, ```sklearn```, ```pandas```, ```seaborn```

### Struttura
- il progetto è strutturato in questa maniera, all'interno della directory ```src```:
  1. ```xxx.py``` description;
  2. ```xxx.py``` description;
- la directory ```data``` ospita il dataset utilizzato, in formato ```.csv```;

### DISCLAIMER
- la struttura del dataset proviene dal sito internet https://www.kaggle.com/adityakadiwal/water-potability con licenza d'uso di tipo **CC0: Public Domain**. Il dataset è da considerarsi utile esclusivamente a fini didattici. I dati in esso contenuti vengono forniti senza alcuna garanzia di affidabilità a proposito della loro accuratezza, adeguatezza e utilità ai fini di una valutazione di sicura potabilità di un'acqua presa in analisi.
