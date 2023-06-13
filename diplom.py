import numpy as np
import sympy as sp
from enum import Enum

class RootSystemGenerator:

    def generate_An(dim):
        res = np.random.randint(0, 1, (dim, dim + 1))
        for i in range(dim):
            res[i, i] = -1
            res[i, i + 1] = 1
        return RootSystem(res)

    def generate_full_An_roots(dim):
        res = []
        for k in range(dim + 1):
            for j in range(dim + 1):
                if j != k:
                    root = np.random.randint(0, 1, dim + 1)
                    root[j] = 1
                    root[k] = -1
                    res.append(root)
        return np.array(res)

    def generate_Bn(dim):
        res = np.random.randint(0, 1, (dim, dim))
        res[0, 0] = 1
        for i in range(1, dim):
            res[i, i - 1] = -1
            res[i, i] = 1
        return RootSystem(res)

    def generate_full_Bn_roots(dim):
        res = []
        for k in range(dim):
            for j in range(k, dim):
                if j != k:
                    root = np.random.randint(0, 1, dim)
                    root[j] = 1
                    root[k] = -1
                    res.append(root)
                    res.append(root * (-1))
                else:
                    root = np.random.randint(0, 1, dim)
                    root[j] = 1
                    res.append(root)
                    res.append(root * (-1))
        for k in range(dim):
            for j in range(k + 1, dim):
                root = np.random.randint(0, 1, dim)
                root[j] = 1
                root[k] = 1
                res.append(root)
                res.append(root * (-1))
        return np.array(res)

    def generate_Dn(dim):
        res = np.random.randint(0, 1, (dim, dim))
        res[0, 0] = 1
        res[0, 1] = 1
        for i in range(1, dim):
            res[i, i - 1] = -1
            res[i, i] = 1
        return RootSystem(res)

    def generate_full_Dn_roots(dim):
        res = []
        for k in range(dim):
            for j in range(k + 1, dim):
                root = np.random.randint(0, 1, dim)
                root[j] = 1
                root[k] = -1
                res.append(root)
                res.append(root * (-1))
        for k in range(dim):
            for j in range(k + 1, dim):
                root = np.random.randint(0, 1, dim)
                root[j] = 1
                root[k] = 1
                res.append(root)
                res.append(root * (-1))
        return np.array(res)

    def generate_point(dim, name):
        a = sp.symbols('{:}:{:}'.format(name, dim))
        u = []
        for i in range(dim):
            u.append(a[i])
        return np.array(u)

    def generate_variables(dim):
        a = sp.symbols('x:{:}'.format(dim))
        x = []
        for i in range(dim):
            x.append(a[i])
        return np.array(x)

    def get_An_orbit(dim):
        a = RootSystemGenerator.generate_point(dim + 1, 'q')
        points = multiset_permutations(a)
        res = []
        for p in points:
            res.append(p)
        return np.array(res)

    @staticmethod
    def generate_sign_changes(dim):
        matrices = []
        for i in range(2**dim):
            bin_number = str(bin(i))[2:]
            sign_change = np.eye(dim, dtype=int)
            for j in range(len(bin_number)):
                if bin_number[len(bin_number) - j - 1] == '1':
                    sign_change[dim - j - 1, dim - j - 1] = -1
            matrices.append(sign_change)
        return np.array(matrices)

    def get_Bn_orbit(dim):
        a = RootSystemGenerator.generate_point(dim, 'q')
        points = multiset_permutations(a)
        permutations = []
        for p in points:
            permutations.append(np.array(p))
        permutations = np.array(permutations)
        sign_changes = RootSystemGenerator.generate_sign_changes(dim)
        res = []
        for i in range(len(permutations)):
            line = []
            for j in range(len(sign_changes)):
                line.append(np.dot(sign_changes[j], permutations[i]))
            res.append(line)
        return res

    def get_Dn_orbit(dim):
        a = RootSystemGenerator.generate_point(dim, 'q')
        points = multiset_permutations(a)
        permutations = []
        for p in points:
            permutations.append(np.array(p))
        permutations = np.array(permutations)
        sign_changes = RootSystemGenerator.generate_sign_changes(dim)
        res = []
        for i in range(len(permutations)):
            line = []
            for j in range(len(sign_changes)):
                if np.linalg.det(sign_changes[j]) > 0:
                    line.append(np.dot(sign_changes[j], permutations[i]))
            res.append(line)
        return res


def get_positive_roots(system_type, dim):
    simple_roots = system_type.generate_simple(dim).simple_roots
    full_roots = system_type.generate_full(dim)
    result = []
    for root in full_roots:
        if np.all(np.array(SystemSolver.solve_by_gauss(simple_roots.T, root).values()) > 0):
            result.append(root)
    return np.array(result)


def get_weight_function(system_type, dim):
    vars_dim = dim + 1 if system_type == RootSystemType.A else dim
    positive_roots = get_positive_roots(system_type, dim)
    variables = RootSystemGenerator.generate_variables(vars_dim)
    return np.prod(variables.dot(positive_roots.T)).expand()


def get_volume_function():
    variables = RootSystemGenerator.generate_variables(1)
    return 1 + 0*variables[0]


def compute(system_type, dim, function):
    point_dim = dim + 1 if system_type == RootSystemType.A else dim
    roots = system_type.generate_simple(dim)
    point = RootSystemGenerator.generate_point(point_dim, 'a')
    return IntegralComputer.compute(roots,
                                    IntegralComputer.point_preprocessing(point, roots),
                                    IntegralComputer.function_preprocessing(function, roots)).expand()


def get_function_in_point(function, point):
    variables = IntegralComputer.get_function_variables(function, 'a')
    sub = []
    for i in range(len(variables)):
        sub.append([variables[i], point[i]])
    return function.subs(sub).expand()


def compute_weight_function_on_orbit_of_An(dim):
    f = get_weight_function(RootSystemType.A, dim)
    print("Integral computing...")
    weights = compute(RootSystemType.A, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_An_orbit(dim)
    a = RootSystemGenerator.generate_point(dim + 1, 'q')
    an = 0
    for i in range(len(a) - 1):
        an -= a[i]
    sum = 0
    for i in range(len(orbit)):
        print(i+1, '/', len(orbit))
        sum += get_function_in_point(weights, orbit[i]).subs(a[len(a) - 1], an)# * (-1) ** i
    return sum.expand()


def compute_weight_function_on_orbit_of_An_signed(dim):
    f = get_weight_function(RootSystemType.A, dim)
    print("Integral computing...")
    weights = compute(RootSystemType.A, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_An_orbit(dim)
    a = RootSystemGenerator.generate_point(dim + 1, 'q')
    an = 0
    for i in range(len(a) - 1):
        an -= a[i]
    sum = 0
    for i in range(len(orbit)):
        print(i+1, '/', len(orbit))
        sum += get_function_in_point(weights, orbit[i]).subs(a[len(a) - 1], an) * (-1) ** i
    return sum.expand()


def compute_weight_function_on_orbit_of_Dn(dim):
    f = get_weight_function(RootSystemType.D, dim)
    print("Integral computing...")
    weights = compute(RootSystemType.D, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_Dn_orbit(dim)
    sum = 0
    iters = len(orbit) * len(orbit[0])
    for i in range(len(orbit)):
        for j in range(len(orbit[i])):
            print(i * len(orbit[0]) + j+1, '/', iters)
            sum += get_function_in_point(weights, orbit[i][j])# * (-1) ** i
    return sum.expand()


def compute_weight_function_on_orbit_of_Dn_signed(dim):
    f = get_weight_function(RootSystemType.D, dim)
    print("Integral computing...")
    weights = compute(RootSystemType.D, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_Dn_orbit(dim)
    sum = 0
    iters = len(orbit) * len(orbit[0])
    for i in range(len(orbit)):
        for j in range(len(orbit[i])):
            print(i * len(orbit[0]) + j+1, '/', iters)
            sum += get_function_in_point(weights, orbit[i][j]) * (-1) ** (i + j)
    return sum.expand()


def compute_weight_function_on_orbit_of_Bn(dim):
    f = get_weight_function(RootSystemType.B, dim)
    print("Integral computing...")
    weights = compute(RootSystemType.B, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_Bn_orbit(dim)
    sum = 0
    iters = len(orbit) * len(orbit[0])
    for i in range(len(orbit)):
        for j in range(len(orbit[i])):
            print(i * len(orbit[0]) + j + 1, '/', iters)
            sum += get_function_in_point(weights, orbit[i][j])# * (-1) ** (i + j)
    return sum.expand()


def compute_weight_function_on_orbit_of_Bn_signed(dim):
    f = get_weight_function(RootSystemType.B, dim)
    print("Integral computing...")
    weights = compute(RootSystemType.B, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_Bn_orbit(dim)
    sum = 0
    iters = len(orbit) * len(orbit[0])
    for i in range(len(orbit)):
        for j in range(len(orbit[i])):
            print(i * len(orbit[0]) + j + 1, '/', iters)
            sum += get_function_in_point(weights, orbit[i][j]) * (-1) ** (i + j)
    return sum.expand()


def compute_volume_on_orbit_of_An(dim):
    f = get_volume_function()
    print("Integral computing...")
    weights = compute(RootSystemType.A, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_An_orbit(dim)
    a = RootSystemGenerator.generate_point(dim + 1, 'q')
    an = 0
    for i in range(len(a) - 1):
        an -= a[i]
    sum = 0
    for i in range(len(orbit)):
        print(i+1, '/', len(orbit))
        sum += get_function_in_point(weights, orbit[i]).subs(a[len(a) - 1], an) #* (-1) ** i
    return sum.expand()


def compute_volume_on_orbit_of_Dn(dim):
    f = get_volume_function()
    print("Integral computing...")
    weights = compute(RootSystemType.D, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_Dn_orbit(dim)
    sum = 0
    iters = len(orbit) * len(orbit[0])
    for i in range(len(orbit)):
        for j in range(len(orbit[i])):
            print(i * len(orbit[0]) + j+1, '/', iters)
            sum += get_function_in_point(weights, orbit[i][j])# * (-1) ** i
    return sum.expand()


def compute_volume_on_orbit_of_Bn(dim):
    f = get_volume_function()
    print("Integral computing...")
    weights = compute(RootSystemType.B, dim, f)
    print("Getting orbit...")
    orbit = RootSystemGenerator.get_Bn_orbit(dim)
    sum = 0
    iters = len(orbit) * len(orbit[0])
    for i in range(len(orbit)):
        for j in range(len(orbit[i])):
            print(i * len(orbit[0]) + j + 1, '/', iters)
            sum += get_function_in_point(weights, orbit[i][j]) #* (-1) ** (i + j)
    return sum.expand()


class RootSystemType(Enum):
    A = (RootSystemGenerator.generate_An, RootSystemGenerator.generate_full_An_roots)
    B = (RootSystemGenerator.generate_Bn, RootSystemGenerator.generate_full_Bn_roots)
    D = (RootSystemGenerator.generate_Dn, RootSystemGenerator.generate_full_Dn_roots)

    def __init__(self, generate_simple, generate_full):
        self.generate_simple = generate_simple
        self.generate_full = generate_full

class RootSystem:

    def __init__(self, roots):
        self.dim = len(roots)
        self.simple_roots = np.array(roots)

    def get_simple_roots(self):
        return self.simple_roots

    def get_gramm_matrix(self):
        G = sp.zeros(self.dim)
        for i in range(self.dim):
            for j in range(i, self.dim):
                G[i, j] = np.dot(self.simple_roots[i], self.simple_roots[j])
                G[j, i] = G[i, j]
        return G

from sympy.utilities.iterables import multiset_permutations
    

class IntegralComputer:

    def compute(root_system, point, function):
        print(root_system.simple_roots)
        if root_system.dim == 1:
            var = IntegralComputer.get_function_variables(
                function, IntegralComputer.get_function_variables_name(function))
            if len(var) != 0:
                return sp.integrate(function, (var, 0, point[0]))
            else:
                return function * point[0]

        result = 0
        roots = root_system.simple_roots
        gramm_matrix = root_system.get_gramm_matrix()

        for k in range(len(roots)):
            A_k = np.delete(roots, k, axis=0)
            delta_k = RootSystem(A_k)
            e_k = sp.zeros(root_system.dim)[:, 0]
            e_k[k] = 1

            v_k = SystemSolver.solve_by_gauss(gramm_matrix, e_k)
            J_k = sp.eye(root_system.dim)
            J_k[:, k] = v_k
            det_J_k = J_k[k, k]

            F_k = IntegralComputer.coordinates_change(J_k, function,
                                                      IntegralComputer.get_function_variables_name(function))
            var_k = sp.Symbol('{}{}'.format(IntegralComputer.get_function_variables_name(F_k), k))
            F_k = IntegralComputer.multipy_Fk_by_t(F_k)

            G_k = gramm_matrix.copy()
            G_k[:, k] = e_k
            point_coords_k = SystemSolver.solve_by_gauss(G_k, np.dot(gramm_matrix, point))
            F_k = F_k.subs(var_k, point_coords_k[k])
            new_point_coords = np.delete(point_coords_k, k)
            t = sp.Symbol('t')
            print("Integrate...")
            f_integ = sp.integrate((F_k * (t ** delta_k.dim)).simplify(), (t, 0, 1))
            print("Recursive...")
            result = result + det_J_k * (point_coords_k[k]) * \
                     IntegralComputer.compute(
                         delta_k, new_point_coords, f_integ)
        return result.simplify()

    def multipy_Fk_by_t(function):
        t = sp.Symbol('t')
        variables = IntegralComputer.get_function_variables(function,
                                                            IntegralComputer.get_function_variables_name(function))
        result_f = function
        for i in range(len(variables)):
            result_f = result_f.subs(variables[i], variables[i] * t)
        return result_f

    def coordinates_change(matrix, function, old_vars_name):
        rows, cols = matrix.shape
        # Определяем имена новых и старых переменных
        new_vars_name = 's'
        if old_vars_name == 's':
            new_vars_name = 'v'

        result_f = function
        old_vars = IntegralComputer.get_function_variables(function, old_vars_name)
        new_vars = sp.symbols('{}:{}'.format(new_vars_name, cols))

        for i in range(len(old_vars)):
            new_coord = 0
            for j in range(len(new_vars)):
                new_coord = new_coord + matrix[i, j] * new_vars[j]
            result_f = result_f.subs(old_vars[i], new_coord)
        return result_f

    # Замена декартовых координат в функции на координаты простых корней
    def function_preprocessing(function, roots):
        A = roots.get_simple_roots()
        x = IntegralComputer.get_function_variables(function, 'x')  # старые переменные
        v = sp.symbols('v:{}'.format(roots.dim))  # новые переменные
        result_f = function
        for j in range(len(x)):
            new_coord = 0
            for i in range(len(v)):
                new_coord = new_coord + A[i, j] * v[i]
            result_f = result_f.subs(x[j], new_coord)
        return result_f

    # Вычисление координат точки U в координатах простых корней
    def point_preprocessing(point, roots):
        A = roots.simple_roots.copy()
        b = point.copy()
        space_dim = len(A[0])
        if roots.dim != space_dim:
            A = np.delete(A.transpose(), space_dim - 1, axis=0)
            b = np.delete(point, space_dim - 1)
        else:
            A = A.transpose()
        return SystemSolver.solve_by_gauss(A, b)

    def get_function_variables(function, variable_name):
        function_symbols = function.free_symbols
        variables = []
        for s in function_symbols:
            if s.name.startswith(variable_name):
                variables.append(s)
        syms = []
        for v in variables:
            syms.append(v.name)
        syms.sort()
        return sp.symbols(syms)

    def get_function_variables_name(function):
        function_symbols = function.free_symbols
        for s in function_symbols:
            if s.name.startswith('x') or s.name.startswith('s') or s.name.startswith('v'):
                return s.name[0]
        return 'No variables!'


class SystemSolver:

    def solve_by_gauss(matrix, right_side):
        A = sp.Matrix(matrix)
        b = sp.Matrix(right_side)
        N = A.cols
        for j in range(0, N - 1):
            if A[j, j] == 0:
                for i in range(j + 1, N):
                    if A[i, j] != 0:
                        SystemSolver.swap_lines(A, b, j, i)
                        break
            for i in range(j + 1, N):
                c_ij = -sp.Rational(A[i, j], A[j, j])
                A[i, :] = A[i, :] + A[j, :] * c_ij
                b[i] = b[i] + b[j] * c_ij

        y = sp.zeros(N)[:, 0]
        y[N - 1] = b[N - 1] / A[N - 1, N - 1]
        for i in range(N - 2, -1, -1):
            y[i] = b[i]
            for j in range(N - 1, i, -1):
                y[i] = y[i] - y[j] * A[i, j]
            y[i] = y[i] / A[i, i]
        return y

    def swap_lines(A, b, j, k):
        tmp_line = A[k].copy()
        tmp_b = b[k]
        A[k] = A[j]
        b[k] = b[j]
        A[j] = tmp_line
        b[j] = tmp_b


"""
n -- ранг системы корней
signed -- знак слагаемых меняется в зависимости от элемента группы Вейля
без signed -- не меняется
"""
n=4

sum_A = compute_weight_function_on_orbit_of_An(n)
print("sum An= ", sum_A)
sum_B = compute_weight_function_on_orbit_of_Bn_signed(n)
print("sum Bn= ", sum_B)
sum_D = compute_weight_function_on_orbit_of_Dn_signed(n)
print("sum Dn= ", sum_D)
print("result:")
print("sum An= ", sum_A)
print("sum Bn= ", sum_B)
print("sum Dn= ", sum_D)

