from scipy.optimize import minimize
import numpy as np
import time
import matplotlib.pyplot as plt
# # ##todo:============================== two dimension and one virabile =====================================
# """take as an example"""
#
# #step1: define the object function
# def object_func(x):
#     x1=x[0]
#     x2=x[1]
#     return x1**2+x1*x2
#
# #step2: define the different constraints
# def equility_constraint(x):
#     x1 = x[0]
#     x2 = x[1]
#     return x1**3+x1*x2-100
#
#
# def inequality_constraint(x):
#     x1 = x[0]
#     x2 = x[1]
#     return x1**3+x1*x2-50
#
# #step3: define the boudndary of the varibales
# bounds_x1=(0,100)
# bounds_x2=(0,100)
#
# #step4: pack the constraints and bounds to list
# constraint1={'type':'eq','fun':equility_constraint}
# constraint2={'type':'ineq','fun':inequality_constraint}
#
# constraint=[constraint1,constraint2]
# bound=[bounds_x1,bounds_x2]# here we pack the defferent bounds of varibales to a list
#
# #step5: set the initial value for te object fucntion:
# x0=[1,1]
#
# #step6: solve our defined optimial questions
# result=minimize(fun=object_func,x0=x0,method='SLSQP',bounds=bound,constraints=constraint)
# print(result)
# print('solved!!!!!.....................................')

# ##summray by [deng]:
# ##method of the 'BFGS' cares less about the constarints and the boudaries, SO it cannot be utilized to solved the optimization with bounds and constraints
# ## instead we can utilzie the method of 'SLSQP'.
#todo: Why? still need to think and solve
#-----------The BFGS cannot handle the large scale data computation and may fail


##todo: ===========================================funny exercise:
## 1============================= test SLSQP and BFGS
# np.random.seed(0)
# K =np.random.normal(size=(100,100))
#
# #step1: def obj
# def object_fun(x):
#     return np.sum((np.dot(K,x-1))**2) + np.sum(x**2)**2
#
# #step2: def constraint
# pass
#
# #step3: def bounds
# pass
#
# #step4: pack the constraints and bounds to lists
# pass
#
# #step5: set the initial value for the obj_fun
# x0=K[0]
#
# #step6: solve our problem
# time0= time.time()
#
# result= minimize(object_fun,x0=x0,method='SLSQP') # SLSQP: succes 0.21 : BFGS: fail 0.55
##-----------The BFGS cannot handle the large scale data computation and may fail
# print(result)
# time2=time.time()
# print('consume time is {}'.format(time2-time0))

## 2============================= test SLSQP and BFGS
# def object_fun(x):
#     x1=x[0]
#     x2=x[1]
#     return np.exp(-1/(0.1*x1**2+x2**2))
#
# x0=[1,1]
#
# result=minimize(fun=object_fun,x0=x0,method='BFGS') ##method 'BFGS' and 'SLSQP' both work well
# print(result)

# 3 =============================== minmizaing the norm of the vector function(test the least-square)##
# #todo: This is what we want, there is the reference.
# # but here he return the vector level input
# def obj_fun(x):
#
#     A=np.arctan(x)
#     B=np.arctan(np.linspace(0,1,len(x)))
#     return A-B
#
# x0=np.zeros(10)
#
# import scipy.optimize as optm
# result=optm.leastsq(func=obj_fun,x0=x0)
# # result=minimize(fun=obj_fun,x0=x0,method='SLSQP')
# print(result)

#4=====================================compare the BFGS with the least-square)##
# def second_object_fun(x):
#     return np.sum(obj_fun(x)**2)
#
# result=minimize(fun=second_object_fun,x0=x0,method='SLSQP')
# print(result)
# ##todo: after this experiments,BFGS works terrible than the SLSQP method. get the worse results.


##5====================================== cureve fiting, not sure whether this can benfit our task
# def f(t,omega,phi):
#     return np.cos(omega*t+phi)
#
# x=np.linspace(0,3,50)
# y=f(x,1.5,1)+0.1*np.random.normal(size=50)
#
# import scipy.optimize as optm
# results= optm.curve_fit(f,x,y)
# print(results)

##todo: here we are targeting at the multi_dimension and multi_varibles optimization, how to solve?

def toVector(x, y):
    res = np.hstack([x.flatten(), y.flatten()])
    return res

def toWZ(vec):
    # assert vec.shape == (2 * 2 * 4,)
    vector_dim = int(len(vec) / 2)
    x_dim = vec[0:vector_dim][:, np.newaxis]
    y_dim = vec[vector_dim:][:, np.newaxis]
    new_vect = np.concatenate((x_dim, y_dim), axis=1)
    return new_vect

def push2onedim(ele):
    x_d = ele[:, 0]
    y_d = ele[:, 1]
    x0 = toVector(x_d, y_d)
    return x0





# def obj_fun(x):
#     ## position term
#     # pos_term=np.sum((x[0]-y_[0])**2)+np.sum((x[num_points]-y_[num_points])**2)
#     pos_term = (x[0] - y_[0]) ** 2 + (x[num_points] - y_[num_points]) ** 2 + (x[-1] - y_[-1])**2 + (x[num_points-1] - y_[num_points-1]) ** 2
#
#     num_N = int(len(x) / 2)
#     x1_x = x[:num_N - 2]
#     x2_x = x[1:num_N - 1]
#     x3_x = x[2:num_N]
#     x1_y = x[num_N:-2]
#     x2_y = x[num_N + 1:-1]
#     x3_y = x[num_N + 2:]
#
#     ## shape term(laplacian coordinate)
#     dx_x=(x1_x+x3_x-2*x2_x)
#     dx_y=(x1_y+x3_y-2*x2_y)
#     # dx_x = ( x2_x-(x1_x + x3_x)/2)
#     # dx_y = ( x2_y-(x1_y + x3_y)/2)
#
#     y1_x = y_[:num_N - 2]
#     y2_x = y_[1:num_N - 1]
#     y3_x = y_[2:num_N]
#     y1_y = y_[num_N:-2]
#     y2_y = y_[num_N + 1:-1]
#     y3_y = y_[num_N + 2:]
#
#     dy_x = (y1_x + y3_x - 2 * y2_x)
#     dy_y = (y1_y + y3_y - 2 * y2_y)
#
#     # dy_x = ( y2_x-(y1_x + y3_x)/2)
#     # dy_y = (y2_y-(y1_y + y3_y)/2)
#     sh_term=np.sum((dx_x-dy_x)**2)+np.sum((dx_y-dy_y)**2)
#
#     ## smooth term
#     # sm_term=np.sum(x1_x+x3_x-2*x2_x)+np.sum(x1_y+x3_y-2*x2_y)
#     values=100*pos_term+ sh_term#+0.01*sm_term #(80 points, 1 poinst)
#     return values

# time1=time.time()
# x0=push2onedim(x_init)
#
# result=minimize(fun=obj_fun,x0=x0,method='SLSQP')
# time2=time.time()
# print('runing time is {}'.format(time2-time1))
# print(result)
# print(y_)

# from matplotlib import pyplot as plt




def optim_dy(x_init,y_init):
    num_points=x_init.shape[0]
    x0 = push2onedim(x_init)
    y_ = push2onedim(y_init)

    def obj_fun(x):
        ## position term
        ## needs the start and edn points
        pos_term = (x[0] - x0[0]) ** 2 + (x[num_points] - x0[num_points]) ** 2 + (x[-1] - x0[-1]) ** 2 + (
                    x[num_points - 1] - x0[num_points - 1]) ** 2


        ##

        num_N = int(len(x) / 2)
        x1_x = x[:num_N - 2]
        x2_x = x[1:num_N - 1]
        x3_x = x[2:num_N]
        x1_y = x[num_N:-2]
        x2_y = x[num_N + 1:-1]
        x3_y = x[num_N + 2:]


        ## shape term(laplacian coordinate)
        dx_x = (x1_x + x3_x - 2 * x2_x)
        dx_y = (x1_y + x3_y - 2 * x2_y)
        # dx_x = ( x2_x-(x1_x + x3_x)/2)
        # dx_y = ( x2_y-(x1_y + x3_y)/2)

        y1_x = y_[:num_N - 2]
        y2_x = y_[1:num_N - 1]
        y3_x = y_[2:num_N]
        y1_y = y_[num_N:-2]
        y2_y = y_[num_N + 1:-1]
        y3_y = y_[num_N + 2:]

        dy_x = (y1_x + y3_x - 2 * y2_x)
        dy_y = (y1_y + y3_y - 2 * y2_y)

        # dy_x = ( y2_x-(y1_x + y3_x)/2)
        # dy_y = (y2_y-(y1_y + y3_y)/2)
        sh_term = np.sum((dx_x - dy_x) ** 2) + np.sum((dx_y - dy_y) ** 2)


        ## self shape term
        x01_x = x0[:num_N - 2]
        x02_x = x0[1:num_N - 1]
        x03_x = x0[2:num_N]
        x01_y = x0[num_N:-2]
        x02_y = x0[num_N + 1:-1]
        x03_y = x0[num_N + 2:]

        dx0_x = (x01_x + x03_x - 2 * x02_x)
        dx0_y = (x01_y + x03_y - 2 * x02_y)

        sf_shape_term=np.sum((dx_x - dx0_x) ** 2)+np.sum((dx_y - dx0_y) ** 2)
        ## smooth term
        # sm_term=np.sum(x1_x+x3_x-2*x2_x)+np.sum(x1_y+x3_y-2*x2_y)
        values = 1 * pos_term + 50*sh_term# +  5*sf_shape_term# +0.01*sm_term #(80 points, 1 poinst)
        return values


    # time1 = time.time()
    result = minimize(fun=obj_fun, x0=x0, method='SLSQP')
    # time2 = time.time()
    # print('runing time is {}'.format(time2 - time1))
    # print(result)
    # print(y_)
    #
    # #plot the curve
    # x0_plot_x, x0_plot_y = x0[0:int(num_points)], x0[int(num_points):]
    # y0_plot_x, y0_plot_y = y_[0:int(num_points)], y_[int(num_points):]
    # r_plot_x, r_plot_y = result['x'][0:int(num_points)], result['x'][int(num_points):]
    #
    # plt.subplot(221)
    # plt.scatter(x0_plot_x, x0_plot_y)
    # plt.plot(x0_plot_x, x0_plot_y)
    # plt.subplot(222)
    # plt.scatter(y0_plot_x, y0_plot_y)
    # plt.plot(y0_plot_x, y0_plot_y)
    # plt.subplot(223)
    # plt.scatter(r_plot_x, r_plot_y)
    # plt.plot(r_plot_x, r_plot_y)
    # plt.show()
    final=toWZ(result['x'])
    return final

if __name__ == '__main__':
    num_points = 30
    x_init = np.ones((num_points, 2))
    y_init = np.random.random((num_points, 2))
    result=optim_dy(x_init,y_init)
    print(result)


