# 📌 Projekty z zakresu Sztucznej Inteligencji

## 🧬 Podstawowy Algorytm Genetyczny – Problem Plecakowy (0–1)

Celem projektu jest implementacja **podstawowego algorytmu genetycznego** umożliwiającego rozwiązanie **problemu plecakowego 0–1**.

### Opis problemu
Każdy przedmiot opisany jest przez:
- **wartość**,
- **wagę**.

Zadaniem algorytmu jest znalezienie takiego **podzbioru przedmiotów**, aby:
- suma wag **nie przekroczyła zadanego limitu**,
- suma wartości była **maksymalna**.

### Założenia
- problem decyzyjny typu **0–1** (przedmiot jest albo wybrany, albo nie),
- reprezentacja chromosomu w postaci **wektora binarnego**,
- wykorzystanie operatorów:
  - selekcji,
  - krzyżowania,
  - mutacji,
- funkcja dopasowania (*fitness*) uwzględniająca ograniczenie wagowe.

---

## 🧠 Zadanie Regresji z Wykorzystaniem Sieci MLP

Celem ćwiczenia było rozwiązanie **zadania regresji** z wykorzystaniem **wielowarstwowej sieci neuronowej typu MLP (Multi-Layer Perceptron)**.

### Dane
- **Zmienne wejściowe:**  
  `x1`, `x2`, `x3`
- **Zmienna wyjściowa:**  
  ciągła wartość `y`

### Podział danych
- zbiór treningowy: **80%**
- zbiór testowy: **20%**

### Analizowane parametry
Przeprowadzono analizę wpływu:
- liczby neuronów w warstwie ukrytej,
- współczynnika uczenia (*learning rate*)

na jakość predykcji modelu.

### Miary jakości
Do oceny jakości modelu zastosowano:
- **współczynnik korelacji Pearsona** \( R \),
- **średni błąd względny**  
  MAPE (*Mean Absolute Percentage Error*).
