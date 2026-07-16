def root_steffensen(f, x0, tolerance=1e-10, max_iterations=100):
    """
    Znajduje pierwiastek równania f(x)=0 metodą Steffensena.

    Parametry:
        f                -- funkcja f(x)
        x0               -- punkt startowy
        tolerance        -- tolerancja błędu
        max_iterations   -- maksymalna liczba iteracji

    Zwraca:
        (root, error, backward_error, function_evaluations, iterations)
    """

    function_evaluations = 0
    x = x0

    for iteration in range(1, max_iterations + 1):
        f_x = f(x)
        function_evaluations += 1

        # Sprawdzenie, czy już trafiliśmy w pierwiastek
        if abs(f_x) < tolerance:
            return x, 0.0, abs(f_x), function_evaluations, iteration - 1

        # Obliczenia pomocnicze
        f_x_plus_fx = f(x + f_x)
        function_evaluations += 1

        denominator = f_x_plus_fx - f_x

        # Unikamy dzielenia przez zero
        if denominator == 0:
            raise ZeroDivisionError("Metoda Steffensena napotkała dzielenie przez zero.")

        # Iteracja Steffensena (Aitken Δ²)
        x_new = x - (f_x ** 2) / denominator

        error = abs(x_new - x)

        # Sprawdzenie tolerancji
        if error < tolerance:
            backward_error = abs(f(x_new))
            function_evaluations += 1
            return x_new, error, backward_error, function_evaluations, iteration

        x = x_new

    # Jeśli nie osiągnięto zbieżności
    backward_error = abs(f(x))
    function_evaluations += 1
    return x, error, backward_error, function_evaluations, max_iterations