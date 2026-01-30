import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # podział danych 80:20.
from sklearn.neural_network import MLPRegressor # sieć MLP do regresji. 
from sklearn.preprocessing import StandardScaler

from kod.main import MAPE # normalizacja wejść (bardzo ważne dla MLP).

# =========================
# USTAWIENIA
# =========================
# włącza/wyłącza rysowanie pionowych odcinków obrazujących błąd dla punktów testowych.
POKAZ_ODCINKI_BLEDU = True   # ustaw False, jeśli odcinki robią wykres nieczytelny
# ogranicza liczbę odcinków, bo przy wielu punktach wykres robi się nieczytelny.
MAX_PUNKTY_ODCINKI = 200     # limit punktów, dla których rysujemy odcinki (żeby nie było "spaghetti")

# =========================
# Wczytanie danych
# =========================
# wczytuje arkusz z dane.xlsx
data = pd.read_excel("dane.xlsx")
# X bierze 3 pierwsze kolumny jako wejścia x1, x2, x3
X = data.iloc[:, :3].values # wejścia X
# Y to kolumna 4 jako wyjście y
Y = data.iloc[:, 3].values # wyjście Y

# Podział 80:20
# podział na zbiór treningowy i testowy 80:20
# random_state=42 zapewnia powtarzalność podziału (te same próbki w teście przy każdym uruchomieniu).
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
) # podział danych 80:20

# Normalizacja
scaler = StandardScaler() # normalizacja wejść
# fit_transform na treningu liczy średnią i odchylenie standardowe dla każdej cechy.
X_train_scaled = scaler.fit_transform(X_train) # normalizacja treningu
# transform na teście używa tych samych parametrów – to eliminuje „wyciek informacji” ze zbioru testowego
X_test_scaled = scaler.transform(X_test) # normalizacja testu

# Konfiguracje (6 modeli)
neurony = [10, 20, 50] # liczby neuronów w warstwie ukrytej
learning_rates = [0.01, 0.05] # współczynniki uczenia

# Liczy MAPE w procentach zgodnie z definicją z instrukcji
# Zabezpieczenie przed dzieleniem przez 0 - żeby uniknąć dzielenia przez zero); wartości takie ignoruje w średniej poprzez nanmean
# Mean Absolute Percentage Error (MAPE)
#            n
#          1 ──     | Y_t,i - Y_p,i |
# MAPE = ────  Σ    | ------------- | · 100%
#          n i=1    |     Y_t,i     |
# Metryka ta określa średnią wartość bezwzględnego błędu predykcji w relacji do wartości rzeczywistej, wyrażoną w procentach.
# Interpretacja MAPE
# MAPE=0% – idealna predykcja,
# niska wartość MAPE – wysoka dokładność ilościowa,
# wysoka wartość MAPE – duże błędy względne.
#
def mape(y_true, y_pred):
    # MAPE w %; w razie Yt=0 zabezpieczenie przed dzieleniem przez 0
    y_true = np.asarray(y_true) # konwersja do tablicy numpy
    y_pred = np.asarray(y_pred) # konwersja do tablicy numpy
    denom = np.where(y_true == 0, np.nan, y_true) # zabezpieczenie przed dzieleniem przez 0
    return np.nanmean(np.abs((y_true - y_pred) / denom)) * 100 # MAPE w %

# --- Konfiguracje ---
# Definicja konfiguracji (6 modeli) do przetestowania
# Każda konfiguracja to para (liczba neuronów, learning rate)
# Różne liczby neuronów w warstwie ukrytej - 3 opcje
# wyniki_koncowe – lista słowników, z których potem robimy DataFrame do Excela (tabela metryk).
wyniki_koncowe = []
# Różne wartości współczynnika uczenia - 2 opcje
# predykcje_modeli – przechowuje predykcje każdego modelu pod kluczem (neurony, lr), żeby później szybko zrobić wykres reszt dla najlepszego modelu.
predykcje_modeli = {} # do przechowania predykcji pod wybór najlepszego modelu

# =========================
# Trenowanie + wykres zgodności dla każdego modelu
# =========================
# Dla każdej konfiguracji tworzy się sieć z jedną warstwą ukrytą: (n,)
# activation="tanh" – funkcja aktywacji
# solver="adam" – algorytm optymalizacji
# max_iter=1000 – maksymalna liczba iteracji (żeby model miał czas się zbiec).
# learning_rate_init = wsp_uczenia – krok uczenia (dla adam ma sens)
# random_state=42 – powtarzalność inicjalizacji i uczenia
for n in neurony:
    for lr in learning_rates:
        warstwa_ukryta = (n,) # liczba neuronów w warstwie ukrytej
        wsp_uczenia = lr # współczynnik uczenia

        # Tworzenie modelu MLP
        MLP_model = MLPRegressor(
            hidden_layer_sizes=warstwa_ukryta, # liczba neuronów w warstwie ukrytej
            activation="tanh", # funkcja aktywacji
            solver="adam", # algorytm optymalizacji
            max_iter=1000, # maksymalna liczba iteracji
            learning_rate_init=wsp_uczenia, # krok uczenia
            random_state=42 # powtarzalność inicjalizacji i uczenia
        )

        # Trenowanie
        # fit uczy model na treningu
        # moment, w którym model uczy się zależności X -> Y
        # metoda .fir(X, Y) uruchamia proces uczenia (trenowania) modelu
        # dopasowuje parametry modelu (wagi i biasy) na podstawie danych treningowych
        # tak, aby minimalizować błąd predykcji na danych treningowych.
        # Po wykonaniu fit obiekt MLP_model przechowuje wyuczone parametry i może wykonywać predykcję przez .predict(...).
        # X_train_scaled - to macierz cech wejściowych po normalizacji (StandardScaler)
        # Y_train - to wektor wartości docelowych (wyjść) dla danych treningowych
        # Predykacja w przód - wejście -> warstwa ukryta (aktywacja tanh) -> wyjście (w regresji wyjście jest liniowe).
        # Learning rate
        # zbyt mały → uczenie wolne, możliwe niedouczenie w ramach max_iter,
        # zbyt duży → niestabilność, przeskakiwanie minimum, gorsze wyniki.
        # Liczba neuronów
        # mało neuronów → mała zdolność aproksymacji (underfitting),
        # dużo neuronów → większa elastyczność, ale większe ryzyko przeuczenia.
        # Instrukcja MLP_model.fit(X_train_scaled, Y_train) przeprowadza proces uczenia sieci MLP na  znormalizowanych danych treningowych, dopasowując wagi i biasy metodą optymalizacji Adam w celu minimalizacji błędu regresji.
        MLP_model.fit(X_train_scaled, Y_train)


        # Predykcja
        # predict liczy wyjście dla próbek testowych
        # Celem tej instrukcji jest wyznaczenie wartości wyjściowych sieci neuronowej dla danych,
        # które nie brały udziału w procesie uczenia (zbioru testowego)
        # Metoda predict realizuje propagację w przód (forward pass) sieci neuronowej
        # z wykorzystaniem wyuczonych parametrów.
        # Predykcja na zbiorze testowym umożliwia:
        # ocenę zdolności generalizacji modelu,
        # obliczenie metryk jakości (R, MAPE),
        # Zmienna y_predicted:
        # jest jednowymiarową tablicą NumPy,
        # ma długość równą liczbie próbek w zbiorze testowym,
        # każdemu elementowi y_predicted[i] odpowiada predykcja dla X_test_scaled[i].
        y_predicted = MLP_model.predict(X_test_scaled)

        # Liczenie metryk dla danego modelu

        # R mierzy liniową zgodność trendu między Y_test a y_pred - R to współczynnik korelacji Pearsona
        # Wartość R spełnia R ∈ [−1, 1].
        # W zadaniach regresji poprawnie działający model powinien charakteryzować się dodatnią i możliwie wysoką wartością R.
        # R=1 – idealna dodatnia korelacja liniowa,
        # R=0 – brak korelacji liniowej,
        # R=−1 – idealna ujemna korelacja liniowa.
        # oblicza macierz korelacji Pearsona pomiędzy przekazanymi wektorami
        # domyślnie traktuje każdy argument jako zmienną losową, a obserwacje jako kolejne elementy wektora
        # zwraca macierz korelacji, z której interesuje nas element [0,1] (korelacja między Y_test a y_pred)
        # macierz korelacji 2x2
        # Instrukcja R = np.corrcoef oblicza współczynnik korelacji Pearsona pomiędzy wartościami rzeczywistymi i przewidywanymi, umożliwiając ocenę zgodności trendu predykcji z danymi testowymi.
        R = np.corrcoef(Y_test, y_predicted)[0, 1] # pierwszy wiersz (0) – odniesienie do zmiennej Y_test, druga kolumna (1) – odniesienie do zmiennej y_predicted

        # MAPE_val mierzy średni błąd względny w %
        MAPE_val = mape(Y_test, y_predicted)

        # I zapisuje do listy
        # ista słowników, z których potem robimy DataFrame do Excela (tabela metryk).
        wyniki_koncowe.append({
            "Neurony": n, # liczba neuronów
            "Learning rate": lr, # współczynnik uczenia
            "R": R, # współczynnik korelacji
            "MAPE [%]": MAPE_val # MAPE w %
        })

        # Zapamiętaj predykcje (do wyboru najlepszego modelu)
        key = (n, lr) # klucz jako para (neurony, lr)
        predykcje_modeli[key] = y_predicted # zapamiętanie predykcji pod kluczem

        # =========================
        # Wykres zgodności: Yt vs Yp + linia y=x + (opcjonalnie) odcinki błędu
        # (dla każdego z 6 modeli)
        # =========================
        plt.figure() # nowa figura
        plt.scatter(Y_test, y_predicted) # punkty Yt vs Yp

        # linia idealnej zgodności y=x
        # Przed narysowaniem linii obliczane są wartości graniczne:
        mn = min(np.min(Y_test), np.min(y_predicted)) # minimalna wartość
        mx = max(np.max(Y_test), np.max(y_predicted)) # maksymalna wartość
        plt.plot([mn, mx], [mn, mx]) # linia y=x

        # opcjonalnie: odcinki błędu od punktu (Yt, Yp) do (Yt, Yt)
        if POKAZ_ODCINKI_BLEDU:
            # ogranicz liczbę odcinków dla czytelności
            idx = np.arange(len(Y_test)) # indeksy wszystkich punktów testowych
            if len(idx) > MAX_PUNKTY_ODCINKI:
                idx = np.random.RandomState(42).choice(idx, size=MAX_PUNKTY_ODCINKI, replace=False) # losowy wybór indeksów

            for i in idx:
                # Rysujemy pionowy odcinek w punkcie (Yt, Yp)
                yt = Y_test[i] # wartość oczekiwana
                yp = y_predicted[i] # wartość z modelu
                plt.plot([yt, yt], [yt, yp]) # odcinek błędu

        plt.xlabel("Wartość oczekiwana (Yt)")
        plt.ylabel("Wartość z modelu (Yp)")
        plt.title(f"Zgodność Yt vs Yp (MLP: {n} neuronów, lr={lr})")
        plt.grid(True)
        plt.show()

# =========================
# Zapis metryk do Excela - To daje plik z 6 wierszami
# =========================
df_wyniki = pd.DataFrame(wyniki_koncowe)
df_wyniki.to_excel("metryki_MLP.xlsx", index=False)

# =========================
# Wybór najlepszego modelu:
# priorytet: najniższe MAPE, potem najwyższe R
# =========================
# sortujesz rosnąco po MAPE (najmniej = najlepiej) <<<
# przy remisie malejąco po R (najwięcej = najlepiej) <<<
# pierwszy wiersz to najlepsza konfiguracja.
df_sorted = df_wyniki.sort_values(by=["MAPE [%]", "R"], ascending=[True, False]).reset_index(drop=True)
best_row = df_sorted.iloc[0]
best_key = (int(best_row["Neurony"]), float(best_row["Learning rate"]))
best_pred = predykcje_modeli[best_key]

print("Najlepszy model:", best_key, "MAPE [%] =", best_row["MAPE [%]"], "R =", best_row["R"])

# =========================
# Wykres reszt dla najlepszego modelu: e_i = Yt - Yp
# =========================
# residuals = Y_test - best_pred

# plt.figure()
# plt.scatter(best_pred, residuals)
# plt.axhline(0)
# plt.xlabel("Predykcja (Yp)")
# plt.ylabel("Reszta (e = Yt - Yp)")
# plt.title(f"Wykres reszt dla najlepszego modelu (MLP: {best_key[0]} neuronów, lr={best_key[1]})")
# plt.grid(True)
# plt.show()


"""
- Kod realizuje wymagane 6 konfiguracji MLP, liczy metryki R i MAPE oraz generuje wykresy zgodności Yt vs Yp z linią y=x.

- Najlepszy model jest wybierany na podstawie minimalnego MAPE, a przy podobnych błędach na podstawie maksymalnego R, po czym dla niego generowany jest wykres reszt e_i = Yt - Yp.

"""