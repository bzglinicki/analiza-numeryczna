def root_newton(f, Df, x0, tolerance=1e-10, max_iterations=100):
    """
    Znajduje pierwiastek równania f(x)=0 metodą Newtona.

    Parametry:
        f               -- funkcja f(x)
        Df              -- pochodna f'(x)
        x0              -- punkt startowy
        tolerance       -- tolerancja błędu
        max_iterations  -- maksymalna liczba iteracji

    Zwraca:
        (root, error, backward_error, function_evaluations, iterations)
    """

    x = x0
    function_evaluations = 0

    for iteration in range(1, max_iterations + 1):
        fx = f(x)
        function_evaluations += 1

        Dfx = Df(x)

        if Dfx == 0:
            raise ZeroDivisionError("Pochodna wynosi zero — metoda Newtona nie może być kontynuowana.")

        # Krok Newtona
        x_new = x - fx / Dfx

        # Szacowanie błędu
        error = abs(x_new - x)

        # Sprawdzenie warunku stopu
        if error < tolerance:
            backward_error = abs(f(x_new))
            function_evaluations += 1
            return x_new, error, backward_error, function_evaluations, iteration

        x = x_new

    # Jeśli nie osiągnięto tolerancji
    backward_error = abs(f(x))
    function_evaluations += 1
    return x, error, backward_error, function_evaluations, max_iterations