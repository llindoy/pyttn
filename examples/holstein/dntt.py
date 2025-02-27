import numpy as np
import matplotlib.pyplot as plt

from pyttn import ntreeBuilder

def gen_lattice(a, b, N):
    res = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            if j % 2 == 0:
                res[i, j, :] = np.array( [ i*a, (j//2)*b, i*N+j])
            else:
                res[i, j, :] = np.array( [ i*a+a/2, (j//2)*b+b/2, i*N+j])
    return res

def partition_points_x(x):
    if x.shape[0] % 2 == 0:
        skip = x.shape[0]//2
    else:
        skip = (x.shape[0]+1)//2
    return x[:skip, :], x[skip:, :]

def partition_points_y(x):
    if x.shape[1] % 2 == 1:
        skip = x.shape[1]//2
    else:
        skip = (x.shape[1]+1)//2
    return x[:, :skip], x[:, skip:]


def partition_and_get_path(x, path = None):
    if path is None:
        path = []

    lists = []

    #partition the two sets of points
    if x.shape[0] > 1:
        xs = [*partition_points_x(x)]
    elif x.shape[1] > 1:
        xs = [*partition_points_y(x)]
    else:
        return [ [path, x[0,0,2]]]


    for i in range(2):
        curr_path = path + [i]
        partition = False

        if(xs[i].shape[1] > 1):
            x2s = [*partition_points_y(xs[i])]
            partition=True
        elif xs[i].shape[0] > 1:
            x2s = [*partition_points_x(xs[i])]
            partition=True
        else:
            lists.append([curr_path, xs[i][0,0,2]])
        
        if partition:
            for j in range(2):
                fpath = curr_path + [j]
                lists = lists + partition_and_get_path(x2s[j], fpath)

    return lists

def plot_points(x):
    for xi in x:
        plt.scatter(xi[:, :, 0], xi[:, :, 1], 2+xi[:,:, 2]/10)


def tree_index_to_site_index(lists):
    return [int(li[1]) for li in lists]


def invert_indexing(inds):
    res = [0 for i in inds]
    for i in range(len(inds)):
        res[inds[i]] = i
    return res

def expand_nodes(inds, N):
    res = []
    for i in inds:
        for j in range(N):
            res.append(i*N+j)
    return res

a = 7.662
b = 6.187
N = 18

chi1 = 32
chi2 = 16


meV = 1

wHs = np.array([30.376128616132444, 43.89040624535055, 55.17296830277117, 66.3315461617586, 72.03481928968552, 117.66100431310078, 132.29113972821762, 144.31760697623739, 146.79729094490128, 151.50869048536262, 165.6428891067467, 168.74249406757656, 174.81771979080304, 175.93357757670177, 180.02505612499718, 184.73645566545855, 189.69582360278628, 192.29949176988333, 200.11049627117455, 204.44994321633632])*meV


gHs = np.array([0.3933532270709312, 0.17210237984542553, 0.5334413182692451, 0.19798241325232202, 0.13433848326899098, 0.13673983920991742, 0.12295596957555018, 0.09118658755455757, 0.11672287532399732, 0.09950086523085824, 0.08859011751360167, 0.2732580976985479, 0.1604403243339793, 0.2092044872594615, 0.17637114567182158, 0.1756552458593672, 0.19750903865539918, 0.3458397031804837, 0.24488133743238563, 0.250213931993282])*meV

kb = 1
Ti = 5

Ts=[150, 200, 250, 300, 350, 400]
T = Ts[Ti]
Js = np.array([
        [89.1, 25.7, -105.3],
        [88.5, 24.8, -104.7],
        [87.9, 24.7, -103.7],
        [87.2, 24.0, -102.6],
        [86.7, 24.1, -101.9],
        [86.1, 23.4, -101.0]
    ])*meV

dVs =np.array([
        [13.2, 21.3, 22.5],
        [15.4, 24.2, 25.9],
        [17.2, 27.0, 28.7],
        [19.0, 29.6, 31.3],
        [20.2, 31.2, 33.5],
        [21.6, 32.9, 35.9]
    ])*meV

        
wp = np.array([5,5,5])*meV
gp = dVs[Ti]*np.tanh(wp/(2*kb*T))/(wp)


x = gen_lattice(a, b, N)
lists = partition_and_get_path(x)
conv = tree_index_to_site_index(lists)
site_to_tree = invert_indexing(conv)

Nvib = 2*(len(wHs)+len(wp))
print(Nvib)
site_to_tree = expand_nodes(site_to_tree, Nvib)
print(site_to_tree)
print(N*N, len(site_to_tree))
exit()

#build the ML-MCTDH tree for the vibrational modes
class chi_step:
    def __init__(self, chimax, chimin, N, degree = 2):
        self.chimin = chimin
        if N%degree == 0:
            self.Nl = int(int(np.log(N)/np.log(degree)))
        else:
            self.Nl = int(int(np.log(N)/np.log(degree))+1)

        self.nx = int((chimax-chimin)//(self.Nl-1))

    def __call__(self, l):
        ret=int((self.Nl-l)*self.nx+self.chimin)
        return ret
tree = ntreeBuilder.mlmctdh_tree([chi2 for i in range(N*N)], 2, chi_step(chi1, chi2, N*N))

#linds = tree.leaf_indices()
#for li in linds:
#    ntreeBuilder.mlmctdh_subtree(
#    tree.at(li).in

#plt.scatter(x[:, :, 0], x[:, :, 1], 1+(np.array()/10))
plt.show()
