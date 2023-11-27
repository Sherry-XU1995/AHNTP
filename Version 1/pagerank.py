import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags


def normal_matrix(a):
    # 对于非对称的矩阵，我们的归一化的方式应该是按照行来进行归一化
    for number in range(0, np.size(a, 0)):
        #获取当前行的元素之和。
        row_sum = a.getrow(number).sum()
        if row_sum == 0:
            continue
        else:
            #遍历当前行中的所有非零元素的列索引。这是利用CSR格式矩阵的特性（indptr和indices）来实现的
            for number_col in a.indices[a.indptr[number]:a.indptr[number + 1]]:
                #对当前行的每个非零元素进行归一化，使得整行的和为1
                a[number, number_col] = a[number, number_col] / row_sum
    return a


def normal_matrix_new(a):
    # 对于对称的矩阵，我们可以使用矩阵相乘的方式来进行归一化，这样的话速度比较快。
    d = a.sum(0)
    d = np.array(d)
    d = d[0]
    dd = list(map(lambda x: 0 if x == 0 else np.power(x, -0.5), d))
    D_matrix = diags(dd, 0)
    C = D_matrix.dot(a)
    C = C.dot(D_matrix)
    a = C
    return a


def motif_times(adjacency_matrix, a, type):
    # 在这里，adjacency_matrix是之前我们得到的原始的邻接矩阵，可以是binary或者weighted
    # a表示的是我们需要做的是几阶的motif
    # type表示的是我们需要做的是哪一个motif

    # 下面根据我们给出的公式来对每一个特定的motif矩阵进行计算，
    # motif矩阵从原来的邻接矩阵得到
    if type == 'M1':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_matrix.dot(U_matrix)
            result_1 = result_1.multiply(U_tran)
            adjacency_matrix = result_1
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    elif type == 'M2':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = B_matrix_spare.dot(U_matrix)
            result_1 = result_1.multiply(U_tran)
            result_2 = U_matrix.dot(B_matrix_spare)
            result_2 = result_2.multiply(U_tran)
            result_3 = U_matrix.dot(U_matrix)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    elif type == 'M3':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            result_1 = B_matrix_spare.dot(B_matrix_spare)
            result_1 = result_1.multiply(U_matrix)
            result_2 = B_matrix_spare.dot(U_matrix)
            result_2 = result_2.multiply(B_matrix_spare)
            result_3 = U_matrix.dot(B_matrix_spare)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    elif type == 'M4':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            result_1 = B_matrix_spare.dot(B_matrix_spare)
            result_1 = result_1.multiply(B_matrix_spare)
            adjacency_matrix = result_1
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    elif type == 'M5':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_matrix.dot(U_matrix)
            result_1 = result_1.multiply(U_matrix)
            result_2 = U_matrix.dot(U_tran)
            result_2 = result_2.multiply(U_matrix)
            result_3 = U_tran.dot(U_matrix)
            result_3 = result_3.multiply(U_matrix)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            adjacency_tran = np.transpose(adjacency_matrix)
            adjacency_matrix = adjacency_matrix + adjacency_tran
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    elif type == 'M6':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_matrix.dot(B_matrix_spare)
            result_1 = result_1.multiply(U_matrix)
            result_2 = B_matrix_spare.dot(U_tran)
            result_2 = result_2.multiply(U_tran)
            result_3 = U_tran.dot(U_matrix)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    elif type == 'M7':
        for i in range(a):
            adjacency_tran = np.transpose(adjacency_matrix)
            B_matrix_spare = adjacency_matrix.multiply(adjacency_tran)
            U_matrix = adjacency_matrix - B_matrix_spare
            U_matrix = np.abs(U_matrix)
            U_tran = np.transpose(U_matrix)
            result_1 = U_tran.dot(B_matrix_spare)
            result_1 = result_1.multiply(U_tran)
            result_2 = B_matrix_spare.dot(U_matrix)
            result_2 = result_2.multiply(U_matrix)
            result_3 = U_matrix.dot(U_tran)
            result_3 = result_3.multiply(B_matrix_spare)
            adjacency_matrix = result_1 + result_2
            adjacency_matrix = adjacency_matrix + result_3
            adjacency_matrix = normal_matrix_new(adjacency_matrix)
            # if len(adjacency_matrix.data) > 0:
            #     print(np.max(adjacency_matrix.data), adjacency_matrix.nnz)
    return adjacency_matrix


def construct_motif(trust_pairs, num_users, times, type, alpha_value):
    spare_array_row = []
    spare_array_col = []
    spare_array_data = []
    for a, b in trust_pairs:
        spare_array_row.append(a)
        spare_array_col.append(b)
        spare_array_data.append(1)
    newspare_array_row = []
    newspare_array_col = []
    counttt = 0
    for nnn in range(len(spare_array_row)):
        # if counttt % 100000 == 0:
        #     print('echo is %d' % counttt)
        counttt += 1
        id1 = spare_array_row[nnn]
        id2 = spare_array_col[nnn]
        newspare_array_row.append(id1)
        newspare_array_col.append(id2)
    adjacency_matrix = csr_matrix((spare_array_data, (newspare_array_row, newspare_array_col)),
                                  shape=(num_users, num_users), dtype=np.float64)
    # 下面的三行代码主要是为了能够让得到的邻接矩阵为binary，如果希望是weighed，请注释
    data_array = adjacency_matrix.data
    lennn = data_array.shape[0]
    adjacency_matrix.data = np.ones((1, lennn), dtype=np.float64)[0]
    # print( adjacency_matrix.shape)
    # print('num none zeros:', adjacency_matrix.nnz)
    result_B = adjacency_matrix.copy()
    result_B = normal_matrix(result_B)
    result_C = motif_times(adjacency_matrix, times, type)
    # alpha_value = 0.99
    result_temp1 = result_B.multiply(alpha_value).tolil()
    result_temp2 = result_C.multiply(1 - alpha_value).tolil()
    result_D = result_temp1 + result_temp2
    result_D = result_D.tocsr()
    # return result_C, entry_unique, result_B, result_D
    return result_D


def binary(a_original):
    a = a_original.copy()
    for number in range(0, np.size(a, 0)):
        # if number % 10000 == 0:
        #     print("echo is %d" % number)
        for number_col in a.indices[a.indptr[number]:a.indptr[number + 1]]:
            a[number, number_col] = 1
    return a


def graphMove(a):
    # 构造转移矩阵, we will construct the transfer matrix
    b = np.transpose(a)
    # b为a的转置矩阵
    c = np.zeros((a.shape), dtype=float)
    for j in range(a.shape[1]):
        # if j % 1000 == 0:
        #     print("echo is %d" % j)
        if b[j].sum() == 0:
            continue
        else:
            c[j] = b[j] / (b[j].sum())
    c = np.transpose(c)
    return c


def graphMove_new(a):
    for number in range(0, np.size(a, 0)):
        # if number % 10000 == 0:
        #     print("echo is %d" % number)
        # row_sum = np.sum(a[number, :].toarray())
        row_sum = a.getrow(number).sum()
        if row_sum == 0:
            continue
        else:
            for number_col in a.indices[a.indptr[number]:a.indptr[number + 1]]:
                a[number, number_col] = a[number, number_col] / row_sum
    a = np.transpose(a)
    return a


def graphMove_newest(a):
    d = a.sum(0)
    d = np.array(d)
    d = d[0]
    dd = map(lambda x: 0 if x == 0 else np.power(x, -0.5), d)
    D_matrix = diags(dd, 0)
    C = D_matrix.dot(a)
    C = C.dot(D_matrix)
    a = C
    a = np.transpose(a)
    return a


def firstPr(c):
    # pr值得初始化
    pr = np.zeros((c.shape[0], 1), dtype=float)
    # 构造一个存放pr值得矩阵
    for i in range(c.shape[0]):
        pr[i] = float(1) / c.shape[0]
    return pr


def pageRank(motif_matrix, p, max_times=30000):
    m = graphMove_new(motif_matrix)
    v = firstPr(m)

    e = np.ones((m.shape[0], 1))
    n = m.shape[0]
    count = 0
    # print(m.shape)
    # print(v.shape)
    # m_spare = csr_matrix(m)
    # we will use the equation to compute the value of pagerank 计算pageRank值
    while count <= max_times:
        # 判断pr矩阵是否收敛,(v == p*dot(m,v) + (1-p)*v).all()判断前后的pr矩阵是否相等，若相等则停止循环
        # if count % 10000 == 0:
        #     print(f"PageRank iter steps {count} ...")
        v = p * m.dot(v) + ((1 - p) / n) * e
        count = count + 1
    return v


def generate_pagerank_weights(train_mask, num_users):
    trust_pairs = [(_[0], _[1]) for _ in train_mask if _[2] == 1]
    motif_weights = []
    for motif_type in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'):
        motif_matrix = construct_motif(trust_pairs, num_users, times=1, type=motif_type, alpha_value=0.88)
        pagerank_weights = pageRank(motif_matrix, p=0.8)  # 浏览当前网页的概率为p,假设p=0.8
        motif_weights.append(pagerank_weights)
    motif_weights = np.concatenate(motif_weights, axis=1)
    return motif_weights
