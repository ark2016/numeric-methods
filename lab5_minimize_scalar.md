
### 🔧 Метод Брента: Комбинация золотого сечения и параболической интерполяции

Метод Брента — это гибридный алгоритм, сочетающий надежность метода золотого сечения с быстрой сходимостью параболической интерполяции. Он не требует вычисления производных функции и работает следующим образом:

1. **Инициализация**:

   * Выбирается интервал `[a, b]`, в котором предполагается наличие минимума функции `f(x)`.
   * Устанавливаются точки `x`, `w`, `v` внутри интервала, где:

     * `x` — текущая точка с наименьшим значением `f(x)`.
     * `w` и `v` — предыдущие точки с наименьшими значениями `f(x)`.

2. **Основной цикл**:

   * **Параболическая интерполяция**:

     * Пытается аппроксимировать функцию параболой, проходящей через точки `x`, `w`, `v`.
     * Вычисляется вершина параболы `u`, которая является кандидатом на минимум.
     * Если `u` находится внутри интервала `[a, b]` и удовлетворяет определенным условиям (например, достаточное уменьшение функции), то `u` принимается как новая точка.
   * **Метод золотого сечения**:

     * Если параболическая интерполяция неудачна или `u` не удовлетворяет условиям, используется метод золотого сечения для выбора новой точки `u` внутри интервала `[a, b]`.
   * **Обновление интервала**:

     * В зависимости от значения `f(u)` обновляется интервал `[a, b]` и точки `x`, `w`, `v`.

3. **Условие остановки**:

   * Алгоритм продолжается до тех пор, пока длина интервала `[a, b]` не станет меньше заданной точности `tol`, или пока не будет достигнуто максимальное количество итераций.

---
