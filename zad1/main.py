import pygame
import random
import sys
import json

# --- 1. STAŁE I PARAMETRY ---

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 950

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
LIGHT_GREEN = (200, 255, 200)


# Parametry Problemu Plecakowego (instancja domyślna)
MAX_WEIGHT = 15  # Maksymalna waga plecaka
ITEMS = [
    {"id": 1, "value": 4, "weight": 12},
    {"id": 2, "value": 2, "weight": 1},
    {"id": 3, "value": 2, "weight": 2},
    {"id": 4, "value": 1, "weight": 1},
    {"id": 5, "value": 10, "weight": 4},
    {"id": 6, "value": 4, "weight": 1},
    {"id": 7, "value": 1, "weight": 1},
    {"id": 8, "value": 7, "weight": 6},
    {"id": 9, "value": 3, "weight": 3},
    {"id": 10, "value": 1, "weight": 1},
]

# Parametry Algorytmu Genetycznego
POP_SIZE = 20           # Rozmiar populacji
MAX_GENERATIONS = 300   # Warunek zatrzymania: maksymalna liczba generacji
PC = 0.8                # Globalne prawdopodobieństwo krzyżowania
PM = 0.2                # Globalne prawdopodobieństwo mutacji
STAGNATION_LIMIT = 50   # Liczba generacji bez poprawy najlepszego fitness


# --- 2. FUNKCJE POMOCNICZE PROBLEMU ---

def generate_new_instance(
    item_count=10, # ilość przedmiotów
    max_weight_range=(10, 30), # zakres maksymalnej wagi plecaka
    value_range=(1, 15), # zakres wartości przedmiotów
    weight_range=(1, 15), # zakres wag przedmiotów
):
    """Generuje nowy losowy zestaw przedmiotów i maksymalną wagę."""
    new_items = [] # lista przedmiotów - słownik z id, value, weight
    for i in range(item_count):
        new_items.append({
            "id": i + 1,
            "value": random.randint(*value_range), # losowa wartość przedmiotu
            "weight": random.randint(*weight_range), # losowa waga przedmiotu
        })

    # Efekt po pętli: new_items to lista np. taka:
    '''
    [
        {"id": 1, "value": 12, "weight": 4},
        {"id": 2, "value": 2,  "weight": 15},
        ...
    ]
    '''
    
    new_max_weight = random.randint(*max_weight_range) # losowa maksymalna waga plecaka

    # Zwracamy nowy zestaw przedmiotów i maksymalną wagę
    # krotka (items, max_weight): lista słowników, int
    # jak odebrać: items, max_w = generate_new_instance()
    return new_items, new_max_weight


# --- 3. FUNKCJE ALGORYTMU GENETYCZNEGO ---

# Funkcja initialize_population generuje początkową populację osobników, 
# z których każdy jest losowym binarnym wektorem długości równej liczbie przedmiotów, 
# reprezentującym decyzję o wyborze poszczególnych elementów do plecaka.
#
# pop_size – liczebność populacji (ile osobników ma powstać),
# item_count – liczba genów w osobniku (czyli liczba przedmiotów w problemie).
# populacja to lista list (osobników), każdy osobnik to lista genów (0 lub 1)
def initialize_population(pop_size, item_count):
    """Tworzy początkową losową populację."""
    return [
        [random.randint(0, 1) for _ in range(item_count)] # tworzy listę długości item_count z losowymi 0 lub 1 - 0 jest w plecaku, 1 nie ma
        for _ in range(pop_size) # każda ieracja to osobnik, _ nie interesuje nas indeks
    ]

# Funkcja calculate_fitness oblicza przystosowanie osobnika jako sumę wartości wybranych przedmiotów, a w przypadku przekroczenia dopuszczalnej wagi stosuje dynamiczną karę kwadratową proporcjonalną do stopnia naruszenia ograniczenia.
def calculate_fitness(chromosome, items, max_weight, penalty_factor=300):
    """
    Oblicza wartość przystosowania (fitness) z dynamiczną karą kwadratową.
    Kara: penalty_factor * (przekroczenie_wagi)^2
    trzeba dobrać penalty_factor, żeby algorytm nie faworyzował zbyt ciężkich plecaków
    """
    # Sumujemy wartości tylko tych przedmiotów, które są w plecaku.
    total_value = sum(g * items[i]["value"] for i, g in enumerate(chromosome)) # suma wartości przedmiotów wybranych do plecaka
    
    # Liczymy wagę tylko tych przedmiotów, które mają gen = 1.
    total_weight = sum(g * items[i]["weight"] for i, g in enumerate(chromosome)) # suma wag przedmiotów wybranych do plecaka
    
    # 
    weight_excess = total_weight - max_weight # ile przekroczono maksymalną wagę plecaka

    # Kara za przekroczenie wagi
    if weight_excess > 0: # jeśli przekroczono wagę, to nakładamy karę
        # Dlaczego kwadrat ? Małe przekroczenia, mała kara. Duże przekroczenia, duża kara.
        return total_value - penalty_factor * (weight_excess ** 2) # kara kwadratowa za przekroczenie wagi
    return total_value


# Zastosowano selekcję metodą koła ruletki, przy czym wartości przystosowania są przesuwane do zakresu dodatniego w celu umożliwienia selekcji przy ujemnych fitnessach wynikających z zastosowanej funkcji kary.
# population – aktualna populacja osobników - lista chromosomów
# fitness_values – odpowiadające im wartości przystosowania
def selection_roulette(population, fitness_values):
    """
    Selekcja metodą koła ruletki.
    Przesuwa fitness tak, aby były dodatnie.
    """
    min_fitness = min(fitness_values) # minimalna wartość fitness w populacji
    # Jeśli minimalny fitness jest ujemny, przesuwamy wszystkie wartości w górę (do wartości dodatnich)
    # Dodajemy 1e-6, żeby:
    # nikt nie miał dokładnie zera,
    # każdy osobnik miał niezerową szansę w ruletce.
    adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values] # przesunięcie fitness, aby były dodatnie
    # Suma przeskalowanych fitnessów - mianownik do normalizacji prawdopodobieństw
    total_adjusted_fitness = sum(adjusted_fitness)

    # Zabezpieczenie awaryjne
    # nie powinno się zdarzyć, bo dodano 1e-6, ale..zabezpieczenie na błędy numeryczne, dziwne dane wejściowe itp.
    if total_adjusted_fitness <= 0:
        # Awaryjnie zwracamy losową populację
        return random.choices(population, k=len(population))

    # Obliczamy prawdopodobieństwa wyboru każdego osobnika
    probabilities = [f / total_adjusted_fitness for f in adjusted_fitness]
    # Wybieramy nową populację na podstawie prawdopodobieństw
    new_population = random.choices(population, weights=probabilities, k=len(population))
    # i zwracamy nową populację po selekcji
    return new_population

# Zastosowano krzyżowanie jednopunktowe, które dla każdej pary rodziców jest wykonywane z prawdopodobieństwem 
# PC, a w przypadku braku krzyżowania potomkowie są kopiami rodziców
# parent1, parent2 – dwa chromosomy rodziców (listy genów)
# pc – prawdopodobieństwo krzyżowania
def crossover(parent1, parent2, pc=PC):
    """Krzyżowanie jednopunktowe."""
    # Długość chromosomu = liczba genów = liczba przedmiotów
    chromosome_length = len(parent1)
    # Sprawdzamy, czy przeprowadzić krzyżowanie
    if random.random() < pc:
        # Wybór punktu krzyżowania
        point = random.randint(1, chromosome_length - 1) # Punkt jest losowany z przedziału [1, długość-1], aby uniknąć pustych potomków
        # Tworzenie potomków przez wymianę genów po punkcie krzyż
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else: # Brak krzyżowania, potomkowie to kopie rodziców
        return parent1[:], parent2[:]


def mutate(chromosome, pm=PM):
    """Mutacja bitowa."""
    mutated_chromosome = []
    for gene in chromosome:
        if random.random() < pm:
            mutated_chromosome.append(1 - gene)  # Zamiana 0 ↔ 1
        else:
            mutated_chromosome.append(gene)
    return mutated_chromosome


def run_ga(
    items,
    max_weight,
    pop_size=POP_SIZE,
    max_generations=MAX_GENERATIONS,
    pc=PC,
    pm=PM,
    penalty_factor=300,
    stagnation_limit=STAGNATION_LIMIT,
):
    """
    Główna pętla Algorytmu Genetycznego.
    Zwraca najlepsze rozwiązanie oraz historię fitness.
    """
    item_count = len(items)
    population = initialize_population(pop_size, item_count)

    best_chromosome = None
    best_fitness = -float('inf')
    best_fitness_generation = 0
    best_fitness_history = []
    stagnation_counter = 0

    # Zakładamy, że jeśli nie będzie wcześniejszego zatrzymania,
    # ostatnia generacja to max_generations - 1
    last_generation = max_generations - 1

    for generation in range(max_generations):
        # 1. Ocena populacji
        fitness_values = [
            calculate_fitness(c, items, max_weight, penalty_factor)
            for c in population
        ]

        current_best_fitness = max(fitness_values)
        current_best_index = fitness_values.index(current_best_fitness)

        # Aktualizacja najlepszego osobnika
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = population[current_best_index][:]
            best_fitness_generation = generation
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Warunek wczesnego zatrzymania (stagnacja)
        if stagnation_counter >= stagnation_limit:
            last_generation = generation
            print(
                f"Zatrzymanie w generacji {generation}: "
                f"Brak poprawy przez {stagnation_limit} generacji."
            )
            break

        # Zapis historii najlepszego fitness w tej generacji
        best_fitness_history.append(current_best_fitness)

        # 2. Selekcja
        selected_population = selection_roulette(population, fitness_values)

        # 3–4. Krzyżowanie + mutacja, z elityzmem
        new_population = []

        # Gwarantowany elityzm – najlepszy z całej historii
        if best_chromosome:
            new_population.append(best_chromosome[:])

        while len(new_population) < pop_size:
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)

            child1, child2 = crossover(parent1, parent2, pc=pc)

            child1 = mutate(child1, pm=pm)
            child2 = mutate(child2, pm=pm)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # 5. Zastąpienie populacji
        population = new_population[:pop_size]

    # --- Wyniki końcowe ---

    if best_chromosome:
        final_items = [
            items[i] for i, gene in enumerate(best_chromosome) if gene == 1
        ]
        final_value = sum(item["value"] for item in final_items)
        final_weight = sum(item["weight"] for item in final_items)

        result = {
            "value": final_value,
            "weight": final_weight,
            "items": final_items,
            "chromosome": best_chromosome,
            "fitness_history": best_fitness_history,
            "best_generation": best_fitness_generation,
            "last_generation": last_generation,
        }
    else:
        result = None

    return result


# --- 4. FUNKCJE WIZUALIZACJI PYGAME ---

def draw_text(screen, text, font, color, x, y):
    """Funkcja pomocnicza do rysowania tekstu."""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


def draw_table_old(screen, font, data, x_start, y_start, column_names):
    """Rysuje prostą tabelę."""
    row_height = 30
    col_widths = [100, 100, 100]
    
    # Nagłówki
    x = x_start
    for i, name in enumerate(column_names):
        pygame.draw.rect(screen, BLUE, (x, y_start, col_widths[i], row_height), 0)
        draw_text(screen, name, font, WHITE, x + 5, y_start + 5)
        x += col_widths[i]

    # Wiersze danych
    y = y_start + row_height
    for item in data:
        x = x_start
        pygame.draw.rect(
            screen, BLACK,
            (x, y, sum(col_widths), row_height),
            1  # Ramka wiersza
        )
        # ID
        draw_text(screen, str(item['id']), font, BLACK, x + 5, y + 5)
        x += col_widths[0]
        
        # Wartość
        draw_text(screen, str(item['value']), font, BLACK, x + 5, y + 5)
        x += col_widths[1]

        # Waga
        draw_text(screen, str(item['weight']), font, BLACK, x + 5, y + 5)
        # x += col_widths[2]  # niepotrzebne

        y += row_height


def draw_table(screen, font, data, x_start, y_start, column_names, highlighted_ids=None):
    """Rysuje prostą tabelę, z opcjonalnym podświetleniem wierszy."""
    if highlighted_ids is None:
        highlighted_ids = set()
    else:
        highlighted_ids = set(highlighted_ids)

    row_height = 30
    col_widths = [100, 100, 100]
    
    # Nagłówki
    x = x_start
    for i, name in enumerate(column_names):
        pygame.draw.rect(screen, BLUE, (x, y_start, col_widths[i], row_height), 0)
        draw_text(screen, name, font, WHITE, x + 5, y_start + 5)
        x += col_widths[i]

    # Wiersze danych
    y = y_start + row_height
    for item in data:
        x = x_start
        row_width = sum(col_widths)

        # Jeśli ID przedmiotu jest w zbiorze zaznaczonych – podświetl tło
        if item['id'] in highlighted_ids:
            pygame.draw.rect(screen, LIGHT_GREEN, (x, y, row_width, row_height), 0)

        # Ramka wiersza
        pygame.draw.rect(screen, BLACK, (x, y, row_width, row_height), 1)

        # ID
        draw_text(screen, str(item['id']), font, BLACK, x + 5, y + 5)
        x += col_widths[0]
        
        # Wartość
        draw_text(screen, str(item['value']), font, BLACK, x + 5, y + 5)
        x += col_widths[1]

        # Waga
        draw_text(screen, str(item['weight']), font, BLACK, x + 5, y + 5)

        y += row_height


def draw_fitness_chart(screen, history, best_gen_index, x_pos, y_pos, width, height, font):
    """Rysuje wykres liniowy ewolucji fitness."""
    if not history:
        draw_text(screen, "Brak danych historii.", font, BLACK, x_pos, y_pos + height // 2)
        return
        
    # Ramka wykresu
    pygame.draw.rect(screen, BLACK, (x_pos, y_pos, width, height), 1)
    draw_text(screen, "Ewolucja Fitness (Generacja vs Wartość)", font, BLACK, x_pos, y_pos - 35)

    max_fit = max(history)
    min_fit = min(history)

    points = []
    for i, fitness in enumerate(history):
        if max_fit == min_fit:
            # Stały fitness → środek wykresu
            y = y_pos + height / 2
        else:
            normalized_y = (fitness - min_fit) / (max_fit - min_fit)
            y = y_pos + height - int(normalized_y * height)

        if len(history) > 1:
            x = x_pos + int((i / (len(history) - 1)) * width)
        else:
            x = x_pos + width // 2

        points.append((x, y))

    # Linia + punkty
    if len(points) > 1:
        pygame.draw.lines(screen, BLUE, False, points, 2)
    for p in points:
        pygame.draw.circle(screen, RED, (int(p[0]), int(p[1])), 2)

    # Oznaczenie najlepszego punktu
    if 0 <= best_gen_index < len(points):
        best_point = points[best_gen_index]
        pygame.draw.circle(screen, GREEN, (int(best_point[0]), int(best_point[1])), 5)

        text = f"Max: {history[best_gen_index]:.1f} (Epoka {best_gen_index})"
        draw_text(screen, text, font, GREEN, int(best_point[0]), int(best_point[1]) - 20)

    # Opisy min/max
    draw_text(screen, f"Max: {max_fit:.1f}", font, BLACK, x_pos + width + 5, y_pos)
    draw_text(
        screen,
        f"Min: {min_fit:.1f}",
        font,
        BLACK,
        x_pos + width + 5,
        y_pos + height - font.get_height(),
    )

def load_items_from_json(path, max_items=10):
    """
    Wczytuje przedmioty z pliku JSON.
    Oczekiwany format: lista obiektów z polami: id, value, weight.
    Zwraca (items, max_weight) lub (None, None) w razie błędu.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Błąd odczytu pliku JSON: {e}")
        return None, None

    if not isinstance(data, list):
        print("Błąd: plik JSON powinien zawierać listę obiektów.")
        return None, None

    items = []
    for i, item in enumerate(data[:max_items]):  # tylko pierwsze max_items
        try:
            _id = int(item.get("id", i + 1))
            value = int(item["value"])
            weight = int(item["weight"])
        except (KeyError, ValueError, TypeError) as e:
            print(f"Pominięto niepoprawny rekord nr {i}: {e}")
            continue

        items.append({
            "id": _id,
            "value": value,
            "weight": weight,
        })

    if not items:
        print("Błąd: po przefiltrowaniu nie ma żadnych poprawnych przedmiotów.")
        return None, None

    # Możesz ustalić MAX_WEIGHT np. jako parametr w pliku,
    # ale jeśli go tam nie ma, to np. 70% sumy wag:
    total_weight = sum(it["weight"] for it in items)
    max_weight = int(0.7 * total_weight) if total_weight > 0 else 10

    return items, max_weight


# --- 5. GŁÓWNA FUNKCJA PROGRAMU ---

def main():
    global ITEMS, MAX_WEIGHT  # jeśli chcesz je później modyfikować globalnie

    # Instancja startowa
    current_items = ITEMS[:]
    current_max_weight = MAX_WEIGHT

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Algorytm Genetyczny - Problem Plecakowy")

    font_small = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)

    clock = pygame.time.Clock()

    # Pierwsze uruchomienie GA
    ga_result = run_ga(current_items, current_max_weight)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Restart GA na tych samych danych
                if event.key == pygame.K_r:
                    print("Restartowanie Algorytmu Genetycznego...")
                    ga_result = run_ga(current_items, current_max_weight)

                # Nowa losowa instancja problemu
                if event.key == pygame.K_n:
                    print("Generowanie nowej instancji problemu...")
                    current_items, current_max_weight = generate_new_instance(item_count=10)
                    ga_result = run_ga(current_items, current_max_weight)
                if event.key == pygame.K_j:
                    print("Próba wczytania przedmiotów z pliku JSON...")
                    json_items, json_max_weight = load_items_from_json("items.json", max_items=10)
                    if json_items is not None:
                        current_items = json_items
                        current_max_weight = json_max_weight
                        ga_result = run_ga(current_items, current_max_weight)
                        print("Pomyślnie wczytano dane z JSON i uruchomiono GA.")
                    else:
                        print("Nie udało się wczytać danych z JSON, pozostają stare dane.")

        screen.fill(WHITE)
        y_cursor = 10

        # Tytuł
        draw_text(screen, "Algorytm Genetyczny: Wyniki Problemu Plecakowego", font_large, BLUE, 50, y_cursor)
        draw_text(
            screen,
            "Naciśnij [R], aby ponownie uruchomić GA, [N] - Nowa Instancja, [J] - wczytaj z JSON.",
            font_small,
            RED,
            50,
            y_cursor + 30,
        )
        y_cursor += 70

        # Sekcja 1: Dane wejściowe
        draw_text(screen, "1. Dane Wejściowe:", font_large, BLACK, 50, y_cursor)
        y_cursor += 30
        draw_text(
            screen,
            f"Maksymalna Waga Plecaka (W_max): {current_max_weight}",
            font_small,
            RED,
            50,
            y_cursor,
        )
        y_cursor += 30

        # Tabela przedmiotów
        # draw_table(screen, font_small, current_items, 50, y_cursor, ["ID", "Wartość (V)", "Waga (W)"])
        # y_cursor += (len(current_items) + 1) * 30 + 20
        # Zbiór ID przedmiotów wybranych przez GA (do podświetlenia)
        if ga_result:
            highlighted_ids = {item['id'] for item in ga_result['items']}
        else:
            highlighted_ids = set()

        # Tabela przedmiotów wejściowych z podświetleniem
        draw_table(
            screen,
            font_small,
            current_items,
            50,
            y_cursor,
            ["ID", "Wartość (V)", "Waga (W)"],
            highlighted_ids=highlighted_ids,
        )
        y_cursor += (len(current_items) + 1) * 30 + 20


        # Sekcja 2: Wyniki algorytmu
        draw_text(screen, "2. Najlepsze Rozwiązanie GA:", font_large, BLACK, 50, y_cursor)
        y_cursor += 30

        if ga_result:
            draw_text(screen, f"Całkowita Wartość: {ga_result['value']}", font_large, GREEN, 50, y_cursor)
            y_cursor += 30
            draw_text(
                screen,
                f"Całkowita Waga: {ga_result['weight']} / {current_max_weight}",
                font_large,
                GREEN,
                50,
                y_cursor,
            )
            y_cursor += 40

            # Lista przedmiotów
            draw_text(screen, "Wybrane Przedmioty:", font_large, BLACK, 50, y_cursor)
            y_cursor += 30

            result_data = [
                {"id": item['id'], "value": item['value'], "weight": item['weight']}
                for item in ga_result['items']
            ]
            draw_table(screen, font_small, result_data, 50, y_cursor, ["ID", "Wartość (V)", "Waga (W)"])
        else:
            draw_text(screen, "Algorytm nie znalazł rozwiązania.", font_large, RED, 50, y_cursor)

        # Sekcja 3: Wykres + komunikat o zatrzymaniu
        if ga_result and 'fitness_history' in ga_result:
            chart_y = max(200, y_cursor - 400)
            draw_fitness_chart(
                screen,
                ga_result['fitness_history'],
                ga_result['best_generation'],
                400,
                chart_y,
                350,
                200,
                font_small,
            )

            # Komunikat o zatrzymaniu przy stagnacji
            font_height = font_small.get_height()
            message_y = SCREEN_HEIGHT - 10 - font_height
            message_x = 50

            if ga_result['last_generation'] < MAX_GENERATIONS - 1:
                stop_message = (
                    f"Zatrzymanie: Konwergencja w generacji {ga_result['last_generation']} "
                    f"(Brak poprawy przez {STAGNATION_LIMIT} generacji)."
                )
                draw_text(screen, stop_message, font_small, BLUE, message_x, message_y)

        pygame.display.flip()
        clock.tick(60)  # ograniczenie do 60 FPS

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()


'''
Klasycznym GA:

Inicjalizacja populacji
Ocena przystosowania
Selekcja
Krzyżowanie
Mutacja
Nowa populacja
Sprawdzenie warunku stopu


'''