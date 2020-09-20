def findMaxSubArray(A):
    """
    Функция возвращает непрерывный подмассив массив ненулевой длины,
    имеющий наибольшую сумму среди всех непрерывных подмассивов массива А.

    Сложность выполнения составляет O(n).
    """

    best_sum = A[0]
    best_start, best_end, current_sum = 0, 0, 0

    for current_end, number in enumerate(A):
        if current_sum <= 0:
            current_start = current_end
            current_sum = number
        else:
            current_sum += number

        if current_sum >= best_sum:
            best_sum = current_sum
            best_start = current_start
            best_end = current_end + 1

    return A[best_start:best_end]
