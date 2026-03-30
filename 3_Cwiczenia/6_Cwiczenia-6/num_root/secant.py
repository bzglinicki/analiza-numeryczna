def root_secant(f, a, b, tolerance=1e-7, max_iterations=100):
    """
    Znajduje przybliżony pierwiastek równania f(x)=0 metodą siecznych
    w przedziale [a, b].

    Zwraca:
        (root, error, backward_error, function_evaluations, iterations)
    """

    fa = f(a)
    fb = f(b)
    function_evaluations = 2

    # Jeśli wartości są równe, metoda nie ruszy
    if fa == fb:
        raise ValueError("f(a) i f(b) nie mogą być równe — metoda siecznych nie zadziała.")

    for iteration in range(1, max_iterations + 1):
        # Wzór metody siecznych
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)
        function_evaluations += 1

        # Obliczenia błędów
        error = abs(x - b)
        backward_error = abs(fx)

        if error < tolerance or backward_error < tolerance:
            return x, error, backward_error, function_evaluations, iteration

        # Przesuwamy punkty
        a, fa = b, fb
        b, fb = x, fx

    # Jeśli nie osiągnięto tolerancji
    return x, error, backward_error, function_evaluations, max_iterations