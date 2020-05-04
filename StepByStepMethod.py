import numpy as np
import math as math


# if user input needs to stored in a list
def get_multiple_input(input_text):
    inputs = []
    while True:
        i = input(input_text)
        try:
            a = int(i)
        except ValueError:
            break
        inputs.append(a)
    return inputs


# if only one value is input
def get_single_input(input_text):
    i = input(input_text)
    while True:
        try:
            i = int(i)
            break
        except ValueError:
            break
    return i


# density of concrete
def get_p_conc():
    p_conc = 2400
    i = input("Enter concrete density (MPa - if not 2400 is default): ")
    if i != "":
        p_conc = int(i)
    return p_conc


# elastic modulus of concrete
def get_E_c(p_conc):
    f_cmi_input = get_single_input("Enter mean in-situ compressive strength (MPa): ")
    if f_cmi_input <= 40:
        E_c = math.pow(p_conc, 1.5) * 0.043 * math.sqrt(f_cmi_input)
    else:
        E_c = math.pow(p_conc, 1.5) * (0.024 * math.sqrt(f_cmi_input) + 0.12)
    return E_c


# variable to calculate the elastic modulus over time
def get_s():
    while True:
        s_input = get_single_input("Enter 0 for Ordinary Portland Cement or 1 for High Early Strength Cement: ")
        if s_input == 0:
            s = 0.38
            break
        elif s_input == 1:
            s = 0.25
            break
        else:
            print("Enter 0 or 1")
    return s


# calculation of alpha2, transfer and service are taken care of in respective loops
def get_alpha2(area, ue):
    th = area / ue
    return 1 + 1.12 * math.exp(-0.008 * th)


# for when th is specifically required
def get_th(area, ue):
    return area / ue


# calculating k2's values at each point
def get_k2_value(alpha2, tao_j, tao_i, th):
    return (alpha2 * math.pow(tao_j - tao_i, 0.8)) / (math.pow(tao_j - tao_i, 0.8) + 0.15 * th)


# arrangement of k2 values
def get_k2(tao_list, transfer, area_transfer, area_service, ue_transfer, ue_service):
    k2 = np.zeros((len(tao_list), len(tao_list)))
    for i in range(0, len(tao_list)):
        for j in range(0, len(tao_list)):
            if i == j: #loading = time
                k2[i, j] = 0

            elif j < i: #time<loading eg Creep(time,loading) = Creep (1,2) = 0
                k2[i, j] = 0

            elif tao_list[j] >= transfer:
                alpha2 = get_alpha2(area_service, ue_service)
                k2[i, j] = get_k2_value(alpha2, tao_list[j], tao_list[i], get_th(area_service, ue_service))

            else:
                alpha2 = get_alpha2(area_transfer, ue_transfer)
                k2[i, j] = get_k2_value(alpha2, tao_list[j], tao_list[i], get_th(area_transfer, ue_transfer))
            j += 1
        i += 1
    return k2


# calculate k4
def get_k4():
    k4_input = get_single_input("Environmental conditions: arid = 0, interior = 1, "
                                "temperate = 2, tropical/coastal = 3: ")
    environment_type = {0: 0.7, 1: 0.65, 2: 0.6, 3: 0.5} #Create a Dictionary for environment types
    return environment_type[k4_input]
    #if k4_input == 0:
    #    k4 = 0.7
    #elif k4_input == 1:
    #    k4 = 0.65
    #elif k4_input == 2:
    #    k4 = 0.6
    #elif k4_input == 3:
    #    k4 = 0.5
    #else:
    #    print("Enter a valid number")
    #return k4


# calculate value of k5 if f_c is between 50 and 100
def get_k5_value(f_c, area, ue):
    alpha2 = get_alpha2(area, ue)
    alpha3 = 0.7 / (k4 * alpha2)
    k5_input = (2 - alpha3) - 0.02 * (1 - alpha3) * f_c
    return k5_input


# k5 conditions
def get_k5(f_c_input, tao_list, transfer, area_transfer, area_service, ue_transfer, ue_service):
    k5 = []
    if f_c_input <= 50:
        k5 = np.ones((1, len(tao_list)))
    elif 50 < f_c_input <= 100:
        for i in range(0, len(tao_list)):
            if tao_list[i] >= transfer:
                k5.append(get_k5_value(f_c_input, area_service, ue_service))

            else:
                k5.append(get_k5_value(f_c_input, area_transfer, ue_transfer))
        k5 = np.array([k5])
    return k5


# calculation and allocation for J_t_tao (creep function = inst + creep strain)
def get_J_t_tao(tao_list, E_c_tao, phi_t_tao):
    J_t_tao = np.zeros((len(tao_list), len(tao_list)))

    for i in range(0, len(tao_list)):
        for j in range(0, len(tao_list)):
            if j < i: # time < loading
                J_t_tao[i, j] = 0
            else: # time >= loading
                J_t_tao[i, j] = (1 / E_c_tao[i]) * (1 + phi_t_tao[i, j])
            j += 1
        i += 1
    return J_t_tao


def get_E_c_j(tao_list, J_t_tao):
    E_c_j = []
    for i in range(0, len(tao_list)):
        for j in range(0, len(tao_list)):
            if i == j:
                E_c_j.append(1 / J_t_tao[i, j])
                j += 1
        i += 1
    return E_c_j


def get_init_N(A_p, E_p, strain_p):
    init_N = A_p * E_p * strain_p
    return init_N


def get_init_M(A_p, E_p, strain_p, y_p, d_ref):
    init_M = A_p * E_p * strain_p * (y_p - d_ref)
    return init_M


def get_f_p_init_N(tao_list, N_array, A_p, E_p, strain_p):
    for i in range(0, len(tao_list)):
        indiv_N = 0
        for j in range(0, len(A_p)):
            indiv_N += get_init_N(A_p[j], E_p, strain_p[j])
        N_array.append(indiv_N)
    return N_array


def get_f_p_init_M(tao_list, M_array, A_p, E_p, strain_p, y_p_s, y_p_t, ref_s, ref_t):
    for i in range(0, len(tao_list)):
        indiv_M = 0
        for j in range(0, len(A_p)):
            if tao_list[i] >= transfer:
                indiv_M += get_init_M(A_p[j], E_p, strain_p[j], y_p_s[j], ref_s)
            else:
                indiv_M += get_init_M(A_p[j], E_p, strain_p[j], y_p_t[j], ref_t)
        f_p_init_M.append(indiv_M)
    return f_p_init_M


def get_B_A_by_y(A_s_list, y_s_list, d_ref): #Used for S and P for sum of total layers
    B_mod = 0
    for i in range(0, len(A_s_list)):
        B_mod += A_s_list[i] * (y_s_list[i] - d_ref)
    return B_mod


def get_I_A_by_y2(A_s_list, y_s_list, d_ref): #Used for S and P for sum of total layers
    I_mod = 0
    for i in range(0, len(A_s_list)):
        I_mod += A_s_list[i] * ((y_s_list[i] - d_ref) ** 2)
    return I_mod


def transform_A(area, mod_steel, A_s_list, mod_pre, A_p_list):
    trans_A = area + (mod_steel - 1) * sum(A_s_list) + (mod_pre - 1) * sum(A_p_list)
    return trans_A


def transform_B(area, c_depth_s, c_depth_t, ref_s, ref_t, mod_steel, tao_list, A_s_list, y_s_list_s, y_s_list_t,
                mod_pre, A_p_list, y_p_list_s, y_p_list_t):
    for i in range(0, len(tao_list)):
        if tao_list[i] >= transfer:
            trans_B = area * (c_depth_s - ref_s) + (mod_steel - 1) * \
                      get_B_A_by_y(A_s_list, y_s_list_s, ref_s) + \
                      (mod_pre - 1) * \
                      get_B_A_by_y(A_p_list, y_p_list_s, ref_s)
        else:
            trans_B = area * (c_depth_t - ref_t) + (mod_steel - 1) * \
                      get_B_A_by_y(A_s_list, y_s_list_t, ref_t) + \
                      (mod_pre - 1) * \
                      get_B_A_by_y(A_p_list, y_p_list_t, ref_t)
    return trans_B


def transform_I(I_gross, area, c_depth_s, c_depth_t, ref_s, ref_t, mod_steel, tao_list, A_s_list, y_s_list, mod_pre,
                A_p_list, y_p_list):
    for i in range(0, len(tao_list)):
        if tao_list[i] >= transfer:
            trans_I = I_gross + area * ((c_depth_s - ref_s) ** 2) + (mod_steel - 1) * get_I_A_by_y2(A_s_list, y_s_list,
                                                                                                    ref_s) + \
                      (mod_pre - 1) * get_I_A_by_y2(A_p_list, y_p_list, ref_s)
        else:
            trans_I = I_gross + area * ((c_depth_t - ref_t) ** 2) + (mod_steel - 1) * get_I_A_by_y2(A_s_list, y_s_list,
                                                                                                    ref_t) + \
                      (mod_pre - 1) * get_I_A_by_y2(A_p_list, y_p_list, ref_t)
    return trans_I


def get_A_c_value(area, A_s_list, A_p_list):
    A = area - sum(A_s_list) - sum(A_p_list)
    return A


def get_A_c(tao_list, area_s, area_t, A_s_list, A_p_list, A_c_array):
    for i in range(0, len(tao_list)):
        if tao_list[i] >= transfer:
            a = get_A_c_value(area_s, A_s_list, A_p_list)
        else:
            a = get_A_c_value(area_t, A_s_list, A_p_list)
        A_c_array.append(a)
    return A_c_array


def get_B_c_value(area, c_depth, d_ref, A_s_list, y_s_list, A_p_list, y_p_list):
    B = area * (c_depth - d_ref) - get_B_A_by_y(A_s_list, y_s_list, d_ref) - get_B_A_by_y(A_p_list, y_p_list, d_ref)
    return B


def get_B_c(tao_list, A_s, A_t, c_depth_s, c_depth_t, ref_s, ref_t,
            A_s_list, y_s_list_s, y_s_list_t, A_p_list, y_p_list_s, y_p_list_t, B_c_array):
    for i in range(0, len(tao_list)):
        if tao_list[i] >= transfer:
            b = get_B_c_value(A_s, c_depth_s, ref_s, A_s_list, y_s_list_s, A_p_list, y_p_list_s)
        else:
            b = get_B_c_value(A_t, c_depth_t, ref_t, A_s_list, y_s_list_t, A_p_list, y_p_list_t)
        B_c_array.append(b)
    return B_c_array


def get_I_c_value(I_gross, area, c_depth, d_ref, A_s_list, y_s_list, A_p_list, y_p_list):
    I = I_gross + (area * (c_depth - d_ref) ** 2) - get_I_A_by_y2(A_s_list, y_s_list, d_ref) - \
        get_I_A_by_y2(A_p_list, y_p_list, d_ref)
    return I


def get_I_c(tao_list, I_gross_s, I_gross_t, A_s, A_t, c_depth_s, c_depth_t, ref_s, ref_t,
            A_s_list, y_s_list_s, y_s_list_t, A_p_list, y_p_list_s, y_p_list_t, I_c_array):
    for i in range(0, len(tao_list)):
        if tao_list[i] >= transfer:
            k = get_I_c_value(I_gross_s, A_s, c_depth_s, ref_s, A_s_list, y_s_list_s, A_p_list, y_p_list_s)
        else:
            k = get_I_c_value(I_gross_t, A_t, c_depth_t, ref_t, A_s_list, y_s_list_t, A_p_list, y_p_list_t)
        I_c_array.append(k)
    return I_c_array


def get_r_b_steel():
    rb_type = {0: (2 / 100), 1: (2.5 / 100), 2: (4 / 100)} #create a dictionary for values
    return rb_type[rb_input]
    #if rb_input == 0:
    #    rb = 0.02
    #elif rb_input == 1:
    #    rb = 0.025
    #elif rb_input == 2:
    #    rb = 0.04
    #else:
    #    print("Enter a valid number")
    #return rb


def get_k5_steel(f_p_steel, sigma_jack):
    gamma = sigma_jack / f_p_steel
    if gamma < 0.4:
        k5_i = 0
    elif 0.4 <= gamma <= 0.7:
        k5_i = (10 * gamma - 4) / 3
    else:
        if r_b_steel == 0.02 or 0.025:
            k5_i = 5 * gamma - 2.5 #low relaxation wire or low relaxation strand
        else:
            k5_i = (50 * gamma) / 6 #Alloy Steel bars
    return k5_i


def get_k6_steel():
    temp = 20
    i = input("Enter average temperature over time period (MPa, 20 degrees celsius is default): ")
    if i != "":
        temp = int(i)
    return temp / 20


# calculation of alpha1, transfer and service are taken care of in respective loops
def get_alpha1(area, ue):
    th = area / ue
    return 0.8 + 1.2 * math.exp(-0.005 * th)


# calculating k1's values at each point
def get_k1_value(alpha1, tao_j, tao_i, th):
    return (alpha1 * math.pow(tao_j - tao_i, 0.8)) / (math.pow(tao_j - tao_i, 0.8) + 0.15 * th)


# k1 values
def get_k1(tao_list, transfer, area_transfer, area_service, ue_transfer, ue_service):
    k1 = np.zeros((1, len(tao_list)))
    for i in range(0, len(tao_list)):
        if tao_list[i] >= transfer:
            alpha1 = get_alpha1(area_service, ue_service)
            k1[0, i] = get_k1_value(alpha1, tao_list[i], transfer - 28, get_th(area_service, ue_service))

        else:
            alpha1 = get_alpha1(area_transfer, ue_transfer)
            k1[0, i] = get_k1_value(alpha1, tao_list[i], 0, get_th(area_transfer, ue_transfer))
    return k1


if __name__ == '__main__':
    # tao = get_multiple_input("Enter days of interest since t = 0 (include day where of first loading): ")
    # N_e_j = get_multiple_input("Enter corresponding applied normal forces for days of interest (in kN): ")
    # M_e_j= get_multiple_input("Enter corresponding applied moment forces for days of interest (in kNm): ")

    tao = [28, 100, 30000]
    transfer = 150
    N_e_j = [-50000, -50000, -50000]
    M_e_j = [100000, 1000000, 100000]
    r_e_j = np.array([[N_e_j], [M_e_j]])

    # p_conc = get_p_conc()
    # E_c = get_E_c(p_conc)
    # s = get_s()
    s = 0.28
    E_c = 30000

    # elastic modulus at each time point
    E_c_tao = []
    for i in tao:
        E_c_tao.append(math.pow(math.exp(s * (1 - math.sqrt(28 / i))), 0.5) * E_c)
    #
    # f_c = get_single_input("Enter concrete characteristic strength (MPa): ")
    # A_t = get_single_input("Enter gross areas at transfer (mm2): ")
    # A_s = get_single_input("Enter gross areas at service (mm2): ")
    # ue_t = get_single_input("Enter exposed perimeter at transfer (mm): ")
    # ue_s = get_single_input("Enter exposed perimeter at service (mm): ")
    #
    f_c = 50
    A_t = 317000
    A_s = 317000
    ue_t = 2200
    ue_s = 2600

    #transfer = 150

    k2 = get_k2(tao, transfer, A_t, A_s, ue_t, ue_s)
    k3 = 2.7 / (1 + math.log10(tao[0]))
    k4 = get_k4()
    i = 0
    k5 = get_k5(f_c, tao, transfer, A_t, A_s, ue_t, ue_s)


    creep_basic_array = np.array(
        [[20, 4.5], [25, 3.8], [32, 3], [40, 2.4], [50, 2], [65, 1.7], [80, 1.5], [100, 1.3]])
    for i in range(len(creep_basic_array)):
        if creep_basic_array[i][0] == f_c:
            phi_basic = creep_basic_array[i][1]
            break
        else:
            if creep_basic_array[i][0] < f_c:
                continue
            else:
                basic_creep_grad = (creep_basic_array[i][1] - creep_basic_array[i - 1][1]) / (
                            creep_basic_array[i][0] - creep_basic_array[i - 1][0])
                phi_basic = creep_basic_array[i - 1][1] + (f_c - creep_basic_array[i - 1][0]) * basic_creep_grad
                break
    phi_t_tao = k2 * k3 * k4 * k5 * phi_basic
    #phi_basic_LT = {20: 4.5, 25: 3.8, 32: 3.0, 40: 2.4, 50: 2, 65: 1.7, 80: 1.5, 100: 1.3}
    #phi_basic = phi_basic_LT[f_c]


    J_t_tao = get_J_t_tao(tao, E_c_tao, phi_t_tao)

    # instantaneous elastic modulus
    E_c_j = get_E_c_j(tao, J_t_tao)

    F_e_j_i = np.zeros((len(tao), len(tao)))

    for i in range(0, len(tao)):
        for j in range(0, len(tao)):
            if i <= j:
                F_e_j_i[j, i] = 0
            else:
                F_e_j_i[j, i] = (J_t_tao[j + 1, i] - J_t_tao[j, i])/ J_t_tao[j, j]
            j += 1
        i += 1

    # first row is all normal forces, second all moments
    #     d_ref_t = get_single_input("Enter distance from top of section to reference axis (mm) at transfer: ")
    #     d_c_t = get_single_input("Enter distance from top of section to centroid of section (mm) at transfer: ")
    #     D_t = get_single_input("Enter overall depth section (mm) at transfer: ")

    #     d_ref_s = get_single_input("Enter distance from top of section to reference axis (mm) at service: ")
    #     d_c_s = get_single_input("Enter distance from top of section to centroid of section (mm) at service: ")
    #     D_slab = get_single_input("Enter slab depth (mm): ")
    #
    #     n_p = get_single_input("Enter modular ratio of pre-stress steel: ")
    #     f_p_A_p = get_multiple_input("Enter total areas of each pre-stress tendon layer (mm2): ")
    #     f_p_E_p = get_single_input("Enter elastic moduli of each pre-stress tendon layer (MPa): ")
    #     f_p_strain_p = get_multiple_input("Enter strain due to pre-stress for each pre-stress tendon layer: ")
    #     f_p_y_p = get_multiple_input("Enter distance from top of section for each pre-stress tendon layer (mm): ")
    #
    #     n_s = get_single_input("Enter modular ratio of steel reinforcement: ")
    #     f_s_A_s = get_multiple_input("Enter total areas of each steel reinforcement layer (mm2): ")
    #     f_s_y_s = get_multiple_input("Enter distance from top of section for each steel reinforcement layer (mm): ")

    #tao = [28, 100, 30000]
    #transfer = 150

    D_t = 1150
    D_slab = 0
    D_s = D_t + D_slab

    A_t = 317000
    A_s = 317000

    ue_t = 2200
    ue_s = 2500

    d_ref_t = 300
    d_ref_s = 300

    d_c_t = 602
    d_c_s = 602

    n_p = 6.25 #how to calculate
    f_p_A_p = [800, 800]
    f_p_E_p = 200000
    f_p_strain_p = [0.00625, 0.00625]
    f_p_y_p_t = [880, 1010]
    f_p_y_p_s = [880, 1010]

    n_s = 6.25 #how to calculate
    f_s_A_s = [900, 1800]
    f_s_y_s_t = [60, 1090]
    f_s_y_s_s = [60, 1090]

    I_gross_t = 49900 * 10 ** 6
    I_gross_s = 49900 * 10 ** 6

    A_j = []
    B_j = []
    I_j = []

    E_c_j = [32000, 36000, 40000]

    r_e_j = np.array([N_e_j, M_e_j])

    f_p_init_N = []
    f_p_init_M = []
    f_p_init_N = get_f_p_init_N(tao, f_p_init_N, f_p_A_p, f_p_E_p, f_p_strain_p)
    f_p_init_M = get_f_p_init_M(tao, f_p_init_M, f_p_A_p, f_p_E_p, f_p_strain_p, f_p_y_p_s, f_p_y_p_t, d_ref_s, d_ref_t)
    f_p_init = np.array([f_p_init_N, f_p_init_M], dtype=np.float64)

for i in (range(0, len(tao))):  # creating 1 x j lists for Aj, Bj, Ij
    if tao[i] >= transfer:
        A_j.append(transform_A(A_s, n_s, f_s_A_s, n_p, f_p_A_p))
        B_j.append(
            transform_B(A_s, d_c_s, d_c_t, d_ref_s, d_ref_t, n_s, tao, f_s_A_s, f_s_y_s_s, f_s_y_s_t, n_p, f_p_A_p,
                        f_p_y_p_s, f_p_y_p_t))
        I_j.append(
            transform_I(I_gross_s, A_s, d_c_s, d_c_t, d_ref_s, d_ref_t, n_s, tao, f_s_A_s, f_s_y_s_s, n_p, f_p_A_p,
                        f_p_y_p_s))
    else:
        A_j.append(transform_A(A_t, n_s, f_s_A_s, n_p, f_p_A_p))
        B_j.append(
            transform_B(A_s, d_c_s, d_c_t, d_ref_s, d_ref_t, n_s, tao, f_s_A_s, f_s_y_s_s, f_s_y_s_t, n_p, f_p_A_p,
                        f_p_y_p_s, f_p_y_p_t))
        I_j.append(
            transform_I(I_gross_t, A_t, d_c_s, d_c_t, d_ref_s, d_ref_t, n_s, tao, f_s_A_s, f_s_y_s_t, n_p, f_p_A_p,
                        f_p_y_p_t))

    F_j_A = []
    F_j_B = []
    F_j_I = []

for i in (range(0, len(tao))):  # creating 1 x j lists for elements in F matrix
    F_j_A.append((1 / (E_c_j[i] * (A_j[i] * I_j[i] - B_j[i] ** 2))) * A_j[i])
    F_j_B.append((1 / (E_c_j[i] * (A_j[i] * I_j[i] - B_j[i] ** 2))) * B_j[i])
    F_j_I.append((1 / (E_c_j[i] * (A_j[i] * I_j[i] - B_j[i] ** 2))) * I_j[i])

F_j = np.array([[F_j_I[0], -F_j_B[0]], [-F_j_B[0], F_j_A[0]]])  # creating initial 2x2 for Fj
for i in range(1, len(tao)):  # appending 2 x 2 arrays
    a = np.array([[F_j_I[i], -F_j_B[i]], [-F_j_B[i], F_j_A[i]]])
    F_j = np.concatenate((F_j, a), axis=1)

    A_c_j = []
    B_c_j = []
    I_c_j = []

    A_c_j = get_A_c(tao, A_s, A_t, f_s_A_s, f_p_A_p, A_c_j)
    B_c_j = get_B_c(tao, A_s, A_t, d_c_s, d_c_t, d_ref_s, d_ref_t,
                f_s_A_s, f_s_y_s_s, f_s_y_s_t, f_p_A_p, f_p_y_p_s, f_p_y_p_t, B_c_j)
    I_c_j = get_I_c(tao, I_gross_s, I_gross_t, A_s, A_t, d_c_s, d_c_t, d_ref_s, d_ref_t,
                f_s_A_s, f_s_y_s_s, f_s_y_s_t, f_p_A_p, f_p_y_p_s, f_p_y_p_t, I_c_j)

    D_c_j = np.array([[A_c_j[0] * E_c_j[0], B_c_j[0] * E_c_j[0]],[B_c_j[0] * E_c_j[0], I_c_j[0] * E_c_j[0]]])

for i in range(1, len(tao)): # appending 2 x 2 arrays
    a = E_c_j[i] * np.array([[A_c_j[i], B_c_j[i]],[B_c_j[i], I_c_j[i]]])
    D_c_j = np.concatenate((D_c_j,a), axis = 1)

f_cr_j = np.zeros((2, len(tao)))
f_sh_j = np.zeros((2, len(tao)))
f_p_rel_j = np.zeros((2, len(tao)))
r_c_j = np.zeros((2, len(tao)))

# time step 1 (tao = 100 days)
# loop for phi_p_init (length = mp)
# rb_input = get_single_input("Type of steel used: low relaxation wire = 0, low relaxation strand = 1, "
#                             "alloy steel bars = 2: ")
# f_p_input = get_single_input("Enter pre-stress tendon breaking strength (MPa): ")
# sigma_p_input = get_single_input("Enter pre-stress jacking stress (MPa): ")

k4_steel = []
for i in range(0,len(tao)):
    k4s = math.log10(5.4 * math.pow(tao[i], 1/6 ))
    k4_steel.append(k4s)

# r_b_steel = get_r_b_steel()
# k5_steel = get_k5_steel(f_p_input, sigma_p_input)
# k6_steel = get_k6_steel()
# k4_steel = np.array(k4_steel, dtype=np.float64)

f_p = 1870 # input: breaking strength
sigma_p_init = 1000 # input: jacking stress
k4_steel = np.array([0.97358677, 1.06572709, 1.47858064],dtype=np.float64)
k5_steel = 0.44919786096256686
k6_steel = 1.25
r_b_steel = 0.02
R_steel = k4_steel * k5_steel * k6_steel * r_b_steel
phi_p_steel = R_steel / (1 - R_steel)


# loop for rel_j
f_p_rel_N = []
f_p_rel_M = []
for i in range(0,len(tao)):
    f_p_rel_N.append(f_p_init_N[i] * phi_p_steel[i])
    f_p_rel_M.append(f_p_init_M[i] * phi_p_steel[i])

f_p_rel_j = np.array([f_p_rel_N, f_p_rel_M], dtype=np.float64)

# fsh loop
# AS5100 shrinkage
A_t = 317000
A_s = 317000
#strain cse
f_c = 40
strain_cse_star = (0.06 * f_c - 1) * 50 * math.pow(10, -6)
strain_cse = []
for i in range(0, len(tao)):
    strain_cse.append(strain_cse_star * (1 - math.exp(-0.1 * tao[i])))


# strain csd
def get_strain_shd_b_star():
    strain_input = get_single_input("Aggregate quality: good = 0, unknown = 1: ")
    if strain_input == 0:
        shd = 800 * math.pow(10, -6)
    elif strain_input == 1:
        shd = 1000 * math.pow(10, -6)
    else:
        print("Enter a valid number")
    return shd
#
# k1 = get_k1(tao, transfer, A_t, A_s, ue_t, ue_s)
# k4 = 0.6
# strain_shd_b_star = get_strain_shd_b_star()
# strain_shd_b = (1 - 0.008 * f_c) * strain_shd_b_star
#
# strain_sh_j = np.array(strain_cse) + strain_shd_b
strain_sh_j = np.array([0, -200 * 10 ** -6, -400 * 10 ** -6]) ## example 5.6 values
#
for i in range(0,len(tao)):
    f_sh_j[0, i] = A_c_j[i] * E_c_j[i] * strain_sh_j[i]
    f_sh_j[1, i] = B_c_j[i] * E_c_j[i] * strain_sh_j[i]

# loop that calculates fcr1 then rc1
# Feji and rcj are set to values here to match book - outputs have been checked already
F_e_j_i = np.array([[0, -1.25, -0.972],[0, 0, -1.777],[0, 0, 0]])
f_p_rel_j = np.array([[ 0, 40000, 60000], [0, 25800000, 38700000]])
r_c_j = np.array([[-1865*10**3, -1465*10**3, 0],[-1165*10**6, -909*10**6, 0]])
r_c_j = np.zeros((f_cr_j.shape[0], len(tao)))
# F_j = np.array([[149.5*10**-12, -181.5*10**-15, 133.6*10**-12, -162.2*10**-15, 120.7*10**-12, -146*10**-15], [-181.5*10**-15, 573.5*10**-18, -162.2*10**-15, 515.4*10**-18, -146*10**-15, 468.1*10**-18]])
#
print(r_e_j)
print(f_cr_j)
print(f_sh_j)
print(f_p_init)
print(f_p_rel_j)
print(F_j)
print(r_c_j)
print(F_e_j_i)

strain_j = np.zeros((2, len(tao)))
m = 0
n = 2
for i in range(0, len(tao)):
    strain_j[:, i] = np.dot(F_j[:, m:n],(r_e_j[:, i] - f_cr_j[:, i] + f_sh_j[:, i]
                                           - f_p_init[:, i] + f_p_rel_j[:, i]))
    r_c_j[:, i] = np.dot(D_c_j[:, m:n], strain_j[:,i]) + f_cr_j[:,i] - f_sh_j[:, i]

    e = 0
    for j in range(0, len(tao)):
        d = F_e_j_i[j, i] * r_c_j[:, i]
        e = e + d
    f_cr_j[:, i] = e
    m += 2
    n += 2

print(r_c_j)
print(f_cr_j)

# c = np.zeros((2,3))
# a = np.array([[1, 2, 3], [1, 2, 3], [1, 2,  3]])
# b = np.array([[10, 10, 10], [20, 20, 20]])
# print(a)
# print(b)
# for i in range(0, len(a)):
#     e = 0
#     for j in range(0, len(a)):
#         d = a[j, i] * b[:,i]
#         e = e + d
#     c[:,i] = e
# print("c:\n", c)

# strain_0 = np.dot(F_j[0:2, 0:2], (r_e_j[:,0] - (f_p_init[:,0])).reshape(-1, 1))

# r_c_j[0:2,0] = (np.dot(D_c_j[0:2,0:2],strain_0) + f_cr_j[0:1,0].reshape(-1, 1) - f_sh_j[0:1,0].reshape(-1, 1)).reshape(1, -1)

