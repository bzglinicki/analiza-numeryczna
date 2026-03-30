def root_bisection(f, a, b, tolerance=1e-7, max_iterations=100):
    """
    Znajduje przybliżony pierwiastek równania f(x)=0 metodą bisekcji
    w przedziale [a, b].

    Zwraca:
        (root, error, backward_error, function_evaluations, iterations)
    """

    fa = f(a)
    fb = f(b)
    function_evaluations = 2

    if fa * fb > 0:
        raise ValueError("Funkcja musi mieć różne znaki na końcach przedziału.")

    for iteration in range(1, max_iterations + 1):
        mid = (a + b) / 2
        fmid = f(mid)
        function_evaluations += 1

        # Sprawdzenie warunku stopu
        error = (b - a) / 2
        backward_error = abs(fmid)

        if error < tolerance or backward_error < tolerance:
            return mid, error, backward_error, function_evaluations, iteration

        # Aktualizacja przedziału
        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    # Jeśli nie osiągnięto tolerancji
    return mid, error, backward_error, function_evaluations, max_iterations