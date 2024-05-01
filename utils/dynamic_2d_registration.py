from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.sparse.linalg import bicg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


## =========================================for image_preprocessing==========================================
def centered_and_normalized(x_matrix, y_matrix):
    x_n_y= np.concatenate((x_matrix,y_matrix),axis=0)
    avg=np.mean(x_n_y,axis=0)
    x_matrix=x_matrix-avg
    y_matrix=y_matrix-avg
    x_n_y = np.concatenate((x_matrix, y_matrix), axis=0)
    r_d= np.sqrt(np.max(np.sum(np.power(x_n_y,2),axis=1)))
    x_matrix=x_matrix/r_d
    y_matrix=y_matrix/r_d
    return x_matrix,y_matrix,avg,r_d


def show_points(x_points, y_points):
    plt.cla()
    plt.plot(x_points[:, 1], x_points[:, 0],'-',color='r')

    plt.scatter(y_points[:, 1], y_points[:, 0], 2, 'b', '*')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()


def compute_normal(y_points):
    R = np.array([[0 ,- 1], [1, 0]])
    y_points = np.matmul(y_points , R.T)
    N1=np.roll(y_points,1,axis=0)-y_points
    n=np.sqrt(np.sum((np.power(N1,2)),axis=1))
    N1[:, 0] = N1[:, 0]/ n
    N1[:, 1] = N1[:, 1]/ n
    N2 =y_points- np.roll(y_points, -1, axis=0)
    n = np.sqrt(np.sum((np.power(N2, 2)), axis=1))
    N2[:, 0] = N2[:, 0] / n
    N2[:, 1] = N2[:, 1] / n
    NY = (N1 + N2) / 2
    n = np.sqrt(np.sum((np.power(NY, 2)), axis=1))
    n[np.where(n==0)]=0.0001
    NY[:, 0] = NY[:, 0] / n
    NY[:, 1] = NY[:, 1] / n
    return NY

## ==========================================for optimization================================================
def optimize_curve_deng(nit_,X_,Y_,Z_,NY_,w3_,w4_):
    w1 = 1
    w2 = 0.1

    ##build M
    M_dim = Z_.shape[0]
    M_row1_ = np.tile(np.arange(M_dim), 2)
    M_col1_1 = np.arange(M_dim)
    M_col1_2 = np.arange(M_dim) + 1
    M_col1_2[-1] = 0
    M_col1_ = np.concatenate((M_col1_1, M_col1_2))
    M1_data_ = np.concatenate([-np.ones(M_dim), np.ones(M_dim)])

    M_col2_1 = M_col1_1
    M_col2_2 = np.arange(M_dim) + 2
    M_col2_2[-2] = 0
    M_col2_2[-1] = 1
    M_col2_ = np.concatenate((M_col2_1, M_col2_2))

    M_row = np.concatenate([M_row1_, M_col1_, M_row1_, M_col2_])
    M_col = np.concatenate([M_col1_, M_row1_, M_col2_, M_row1_])
    M_data = np.concatenate([M1_data_, M1_data_, M1_data_, M1_data_])
    M_ = csr_matrix((M_data, (M_row, M_col)), shape=(4 * M_dim, M_dim))  # .toarray()

    Ex_ = M_ * X_

    ##########################
    #Create kd - tree
    #todo kd = KDTreeSearcher(Y)
    kd = KDTree(Y_)
    ##########################
    #Initialize linear system | | D ^ 0.5(Av - b) | | _2 ^ 2 + | | W ^ 0.5v | | _2 ^ 2
    dim = Z_.shape[0] * Z_.shape[1]
    z_s=Z_.shape[0]

    A_data0=np.ones(dim)
    A_row0 = np.arange(z_s, (z_s + dim))
    A_col0 = np.arange(0, dim)

    A_data1=np.ones(dim)
    A_row1 =np.arange(z_s+dim,z_s+2*dim)
    A_col1 =np.arange(0,dim)

    A_data2=-np.ones((int(dim/2)))
    A_row2=np.arange((z_s+dim),(z_s+dim+int(dim/2)))
    A_col2=np.tile(([dim+1]),(int(dim/2)))

    A_data3=A_data2
    A_row3=np.arange((z_s+dim+int(dim/2)),(z_s+2*dim))
    A_col3=np.tile(([dim+2]),(int(dim/2)))

    A_data4=M_data
    A_row4=M_row+z_s+2*dim#np.arange((z_s+2*dim),(z_s+4*dim))
    A_col4=M_col+0#np.arange(0,z_s)

    A_data5=M_data
    A_row5 = M_row+z_s + 4 * dim #np.arange((z_s + 4 * dim),(z_s+6*dim))
    A_col5 = M_col + z_s #np.arange(z_s, dim)

    A_data=np.concatenate([A_data0,A_data1,A_data2,A_data3,A_data4,A_data5])
    A_row=np.concatenate([A_row0,A_row1,A_row2,A_row3,A_row4,A_row5])
    A_col=np.concatenate([A_col0,A_col1,A_col2,A_col3,A_col4,A_col5])

    # A = sp.coo_matrix((A_data,(A_row,A_col)),shape=(z_s+6*dim, dim+z_s+3)).toarray()
    # A=csr_matrix((z_s+6*dim, dim+z_s+3))#.toarray()
    # A[z_s:(z_s+dim),0:dim]=np.eye(dim,dim)
    # A[(z_s+dim):(z_s + 2*dim), 0:dim] = np.eye(dim,dim)
    # A[(z_s+dim):(z_s+dim+int(dim/2)),dim+1] = -np.ones((int(dim/2)))
    # A[(z_s+dim+int(dim/2)):(z_s+2*dim),dim+2]=-np.ones(int(dim/2))
    # A[(z_s+2*dim):(z_s+4*dim),0:z_s]=M_
    # A[(z_s+4*dim):,z_s:dim]=M_
    # flag = (A == A_).all()

    D_data=np.concatenate((w1*np.ones(z_s),w2*np.ones(dim)))
    D_row=np.arange(0,z_s+dim)
    D_col=np.arange(0,z_s+dim)
    # D = sp.coo_matrix((D_data, (D_row, D_col)), shape=(z_s + 6 * dim, z_s + 6 * dim)).toarray()

    # D=csr_matrix((z_s+6*dim,z_s+6*dim)).toarray()
    # D[0:z_s,0:z_s]=w1*np.eye(z_s,z_s)
    # D[z_s:(z_s+dim),z_s:(z_s+dim)]=w2*np.eye(dim,dim)
    # flag = (D == D_).all()

    W_data= np.concatenate((0.1*np.ones(3),1*np.ones(z_s)))
    W_row=np.arange(dim,dim+z_s+3)
    W_col=np.arange(dim,dim+z_s+3)
    W=csr_matrix((W_data,(W_row,W_col)),shape=(dim+z_s+3,dim+z_s+3))

    # W=csr_matrix((dim+z_s+3,dim+z_s+3)).toarray()
    # W[dim:(dim+3),dim:(dim+3)]=0.1*np.eye(3,3)
    # W[(dim+3):,(dim+3):]=1*np.eye(z_s,z_s)
    # flag = (W == W_).all()

    for it in range(nit_):
        ################################################################
        # kd - tree look - up
        _,idz = kd.query(Z_, k=1)
        P = Y_[idz,:]
        NP = NY_[idz,:]
        ################################################################
        # Build linear system
        N_data = np.concatenate((NP[:, 0], NP[:, 1]))
        N_row = np.tile(np.arange(0, NP.shape[0]), 2)
        N_col = np.arange(0, 2 * NP.shape[0])
        N = csr_matrix((N_data, (N_row, N_col)), shape=(NP.shape[0], 2 * NP.shape[0]))  # .toarray()
        # N1=csr_matrix((NP[:, 0],(np.arange(NP.shape[0]),np.arange(NP.shape[0]))),shape=(NP.shape[0],NP.shape[0])).toarray()
        # N2 = csr_matrix(( NP[:, 1], (np.arange(NP.shape[0]), np.arange(NP.shape[0]))), shape=(NP.shape[0], NP.shape[0])).toarray()
        # N=np.concatenate((N1,N2),axis=1)
        # flag=(N==N_).all()

        ## reconstruct A
        ## we have A_data before
        A_N_data = N_data
        A_N_row = N_row + 0
        A_N_col = N_col + 0
        Xr = X_.copy()
        Xr[:, 0] = -Xr[:, 0]
        Xr = np.fliplr(Xr)
        A_Xr_data = Xr.reshape((dim, 1), order='F').flatten()
        A_Xr_row = np.arange(0, dim) + z_s + dim
        A_Xr_col = np.tile(0, dim) + dim
        A_Ex_data1 = Ex_[:, 1]
        A_Ex_row1 = np.arange(z_s + 2 * dim, z_s + 4 * dim)
        A_Ex_col1 = np.tile(np.arange((dim + 3), dim + z_s + 3), 4)
        A_Ex_data2 = -Ex_[:, 0]
        A_Ex_row2 = np.arange(z_s + 4 * dim, z_s + 6 * dim)
        A_Ex_col2 = np.tile(np.arange((dim + 3), dim + z_s + 3), 4)
        A_data_all = np.concatenate((A_data, A_N_data, A_Xr_data, A_Ex_data1, A_Ex_data2))
        A_row_all = np.concatenate((A_row, A_N_row, A_Xr_row, A_Ex_row1, A_Ex_row2))
        A_col_all = np.concatenate((A_col, A_N_col, A_Xr_col, A_Ex_col1, A_Ex_col2))
        A = sp.coo_matrix((A_data_all, (A_row_all, A_col_all)), shape=(z_s + 6 * dim, dim + z_s + 3))

        # A[0:z_s, 0:dim] = N.toarray()
        # Xr = X_.copy()
        # Xr[:, 0] = -Xr[:, 0]
        # Xr = np.fliplr(Xr)
        # A[(z_s + dim):(z_s + 2 * dim), dim] = Xr.reshape((dim, 1), order='F').flatten()
        # A[(z_s + 2 * dim):3 * dim, (dim + 3):] = csr_matrix((Ex_[0:z_s, 1], (np.arange(z_s), np.arange(z_s))), shape=(z_s, z_s)).toarray()
        # A[3 * dim:(z_s + 3 * dim), (dim + 3):] = csr_matrix((Ex_[z_s:2 * z_s, 1], (np.arange(z_s), np.arange(z_s))),shape=(z_s, z_s)).toarray()
        # A[(z_s + 3 * dim):4 * dim, (dim + 3):] = csr_matrix((Ex_[2 * z_s:3 * z_s, 1], (np.arange(z_s), np.arange(z_s))),shape=(z_s, z_s)).toarray()
        # A[4 * dim:(z_s + 4 * dim), (dim + 3):] = csr_matrix((Ex_[(3 * z_s):, 1], (np.arange(z_s), np.arange(z_s))), shape=(z_s, z_s)).toarray()
        # A[(z_s + 4 * dim):5 * dim, (dim + 3):] = csr_matrix((-Ex_[0:z_s, 0], (np.arange(z_s), np.arange(z_s))),shape=(z_s, z_s)).toarray()
        # A[5 * dim:(z_s + 5 * dim), (dim + 3):] = csr_matrix((-Ex_[z_s:2 * z_s, 0], (np.arange(z_s), np.arange(z_s))),shape=(z_s, z_s)).toarray()
        # A[(z_s + 5 * dim):6 * dim, (dim + 3):] = csr_matrix( (-Ex_[2 * z_s:3 * z_s, 0], (np.arange(z_s), np.arange(z_s))), shape=(z_s, z_s)).toarray()
        # A[(6 * dim):, (dim + 3):] = csr_matrix((-Ex_[3 * z_s:, 0], (np.arange(z_s), np.arange(z_s))),shape=(z_s, z_s)).toarray()
        # flag=(A==A_).all()

        b = np.concatenate((N * P.reshape((dim, 1), order='F'), P.reshape((dim, 1), order='F'),X_.reshape((dim, 1), order='F'), Ex_.reshape((dim * 4, 1), order='F')))
        # b[0:z_s]=np.matmul(N,P.reshape((dim,1),order='F'))
        # b[z_s:(z_s+dim)]=P.reshape((dim,1),order='F')
        # b[(z_s+dim):(z_s+2*dim)]=X_.reshape((dim,1),order='F')
        # b[(z_s+2*dim):]=Ex_.reshape((dim*4,1),order='F')

        ##we have D_data before
        D_data_all=np.concatenate((D_data,w3_*np.ones(dim),w4_*np.ones(dim*4)))
        D_row_all_1=np.arange((z_s+dim),(z_s+2*dim))
        D_col_all_1=np.arange((z_s+dim),(z_s+2*dim))
        D_row_all_2=np.arange((z_s+2*dim),z_s + 6 * dim)
        D_col_all_2=np.arange(z_s+2*dim,z_s + 6 * dim)
        D_row_all=np.concatenate((D_row,D_row_all_1,D_row_all_2))
        D_col_all=np.concatenate((D_col,D_col_all_1,D_col_all_2))
        D=sp.coo_matrix((D_data_all,(D_row_all,D_col_all)),shape=(z_s + 6 * dim,z_s + 6 * dim))#.toarray()
        # D[(z_s + dim):(z_s + 2 * dim), (z_s + dim):(z_s + 2 * dim)] = w3_ * np.eye(dim, dim)
        # D[(z_s + 2 * dim):, (z_s + 2 * dim):] = w4_ * np.eye(dim * 4, dim * 4)
        # flag=(D==D_).all()

        A_csr = A.tocsr()
        D_csr = D.tocsr()
        ###############################################################
        # Solve
        # ! solve linear system
        # v=(sp.linalg.inv(A_csr.transpose()*D*A+W)*A_csr.transpose()*D_csr*b)[:,0]
        v, _ = bicg(A_csr.transpose() * D_csr * A_csr + W, A_csr.transpose() * D_csr * b)
        # v=np.matmul(np.linalg.inv(A.T.dot(D).dot(A)+W), A.T.dot(D).dot(b))[:,0]
        # v, _ = bicg(A.T.dot(D).dot(A) + W, A.T.dot(D).dot(b))
        ###############################################################
        # Extract solution
        Z_=v[0:dim].reshape((X_.shape[0],X_.shape[1]),order='F')
        theta=v[dim]

        R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        X_=np.matmul(X_,R.T)+np.tile(v[dim+1:dim+3].T,(z_s,1))

        # a2 = np.reshape(v[dim + 6:], (-1, 3), 'F')
        # Rset = Rotation.from_euler('zyx', a2[Ni, :]).as_dcm()
        # Ex = np.einsum('ijk,ik->ij', Rset, Ex)

        # Ri = np.reshape(x[dim + 6:], (-1, 3), 'F')
        # Ri = Rotation.from_euler('zyx', Ri[Ni, :]).as_matrix()
        # Ex = np.einsum('ijk, ik->ij', Ri, Ex)
        # test=v[dim+3:dim+3+z_s]
        # Ri=  np.reshape[v[dim+3:z_s]]

        for i in range(z_s):
            theta=v[dim+3+i]
            R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            Ex_[i,:]=np.matmul(Ex_[i,:],R.T)
            Ex_[i + z_s, :] = np.matmul(Ex_[i + z_s,:], R.T)
            Ex_[i + 2*z_s, :] = np.matmul(Ex_[i + 2*z_s, :], R.T)
            Ex_[i + 3 * z_s, :] = np.matmul(Ex_[i + 3 * z_s, :], R.T)

        ###############################################################
        # Show result
        # show_points(Z_, Y_)
        ###############################################################
        # Stopping Criteria
        w3_ = w3_ * 0.4
        w4_ = w4_ * 0.9
        ###############################################################
    return Z_,idz

def reproject(Z_,Y_,avg_,d_):
    Z_=Z_*d_
    Y_=Y_*d_
    Z_=Z_+ np.tile(avg_,(Z_.shape[0],1))
    Y_=Y_+np.tile(avg_,(Y_.shape[0],1))

    return Z_,Y_


def dynamic_2d_registration_deng(x_set_points,y_set_points):
    ## 1. centered and normalized
    X, Y, avg, r_d = centered_and_normalized(x_set_points, y_set_points)
    # show_points(X,Y)
    ## compute normals of Y
    NY = compute_normal(Y)

    ## 4. optimize the curve deformation
    # some hyperparams
    nit = 12
    w3 = 100.0  # global rigid
    w4 = 0.9# the smaller this value, the less snapping
    # utlize Z to replace X
    Z = X
    #solved
    Z, idx = optimize_curve_deng(nit, X, Y, Z, NY, w3, w4)
    # Z, Y = reproject(Z, Y, avg, r_d)
    # show_points(Z,Y)
    return Z,idx


if __name__ == '__main__':
    ## 1. read test data
    with open('../../p_searchInte_module/query_arc.pkl', 'rb') as f:
        query = pickle.load(f)
    f.close()
    with open('../../p_searchInte_module/top.pkl', 'rb') as f:
        top = pickle.load(f)
    f.close()

    ## 2. centered and normalized
    X, Y, avg, r_d= centered_and_normalized(query,top.T)
    show_points(X,Y)

    ## 3. compute normals of Y
    NY = compute_normal(Y)

    ## 4. optimize the curve deformation
    # some hyperparams
    nit = 60
    type = 2
    w3 = 100.0# global rigid
    w4 = 1 # local rigid

    # utlize Z to replace X
    Z = X

    # optimize
    time1=time.time()
    Z,idx = optimize_curve_deng(nit,X,Y,Z,NY,type,w3,w4)
    time2=time.time()

    Z,Y=reproject(Z,Y,avg,r_d)
    show_points(Z,Y)
    print(time2-time1)





