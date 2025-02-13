import numpy as np 
import matplotlib.pyplot as plt 
from itertools import cycle
cycol = cycle('bgrcmk')

class DataGeneratorXYZT:
  """ Generates 3D time-dependent collocation grid for training PINNs based on SciANN's data generator
  
  # Arguments:
    X: [X0, X1]
    Y: [Y0, Y1]
    Z: [Z0, Z1]
    T: [T0, T1]
    targets: list and type of targets you wish to impose on PINNs. 
        ('domain', 'ic', 'bc-left', 'bc-right', 'bc-front', 'bc-back', 'bc-bottom', 'bc-top', 'all')
    num_sample: total number of collocation points. 
    logT: generate random samples logarithmic in time. 

  # Examples: 
    >> dg = DataGeneratorXYZT([0., 1.], [0., 1.], [0., 1.], [0., 1.], 
                             ["domain", "ic", "bc-left", "bc-right", "bc-front", "bc-back", "bc-bottom", "bc-top"], 
                             10000)
    >> input_data, target_data = dg.get_data()

  """
  def __init__(self, 
               X=[0., 1.],
               Y=[0., 1.],
               Z=[0., 1.],
               T=[0., 1.],
               targets=['domain', 'ic', 'bc-left', 'bc-right', 'bc-front', 'bc-back', 'bc-bottom', 'bc-top'], 
               num_sample=10000,
               logT=False):
    'Initialization'
    self.Xdomain = X
    self.Ydomain = Y
    self.Zdomain = Z
    self.Tdomain = T
    self.logT = logT
    self.targets = targets
    self.num_sample = num_sample
    self.input_data = None
    self.target_data = None
    self.set_data()

  def __len__(self):
    return self.input_data[0].shape[0]

  def set_data(self):
    self.input_data, self.target_data = self.generate_data()

  def get_data(self):
    return self.input_data, self.target_data

  def generate_uniform_T_samples(self, num_sample):
    if self.logT is True:
      t_dom = np.random.uniform(np.log1p(self.Tdomain[0]), np.log1p(self.Tdomain[1]), num_sample)
      t_dom = np.exp(t_dom) - 1.
    else:
      t_dom = np.random.uniform(self.Tdomain[0], self.Tdomain[1], num_sample)
    return t_dom

  def generate_data(self):
    # Half of the samples inside the domain.
    num_sample = int(self.num_sample/2)
    
    counter = 0
    # domain points 
    x_dom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    y_dom = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample)
    z_dom = np.random.uniform(self.Zdomain[0], self.Zdomain[1], num_sample)
    t_dom = self.generate_uniform_T_samples(num_sample)
    ids_dom = np.arange(x_dom.shape[0])
    counter += ids_dom.size

    # The other half distributed equally between BC and IC.
    num_sample = int(self.num_sample/4)

    # initial conditions
    x_ic = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample)
    y_ic = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample)
    z_ic = np.random.uniform(self.Zdomain[0], self.Zdomain[1], num_sample)
    t_ic = np.full(num_sample, self.Tdomain[0])
    ids_ic = np.arange(x_ic.shape[0]) + counter 
    counter += ids_ic.size

    # bc points 
    num_sample_per_edge = int(num_sample/6)
    # left bc points 
    x_bc_left = np.full(num_sample_per_edge, self.Xdomain[0])
    y_bc_left = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    z_bc_left = np.random.uniform(self.Zdomain[0], self.Zdomain[1], num_sample_per_edge)
    t_bc_left = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_left = np.arange(x_bc_left.shape[0]) + counter
    counter += ids_bc_left.size

    # right bc points 
    x_bc_right = np.full(num_sample_per_edge, self.Xdomain[1])
    y_bc_right = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    z_bc_right = np.random.uniform(self.Zdomain[0], self.Zdomain[1], num_sample_per_edge)
    t_bc_right = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_right = np.arange(x_bc_right.shape[0]) + counter 
    counter += ids_bc_right.size

    # front bc points 
    x_bc_front = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample_per_edge)
    y_bc_front = np.full(num_sample_per_edge, self.Ydomain[0])
    z_bc_front = np.random.uniform(self.Zdomain[0], self.Zdomain[1], num_sample_per_edge)
    t_bc_front = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_front = np.arange(x_bc_front.shape[0]) + counter
    counter += ids_bc_front.size

    # back bc points 
    x_bc_back = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample_per_edge)
    y_bc_back = np.full(num_sample_per_edge, self.Ydomain[1])
    z_bc_back = np.random.uniform(self.Zdomain[0], self.Zdomain[1], num_sample_per_edge)
    t_bc_back = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_back = np.arange(x_bc_back.shape[0]) + counter 
    counter += ids_bc_back.size

    # bottom bc points 
    x_bc_bottom = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample_per_edge)
    y_bc_bottom = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample_per_edge)
    z_bc_bottom = np.full(num_sample_per_edge, self.Zdomain[0])
    t_bc_bottom = self.generate_uniform_T_samples(num_sample_per_edge)
    ids_bc_bottom = np.arange(x_bc_bottom.shape[0]) + counter
    counter += ids_bc_bottom.size

    # top bc points 
    x_bc_top = np.random.uniform(self.Xdomain[0], self.Xdomain[1], num_sample - num_sample_per_edge)
    y_bc_top = np.random.uniform(self.Ydomain[0], self.Ydomain[1], num_sample - num_sample_per_edge)
    z_bc_top = np.full(num_sample - num_sample_per_edge, self.Zdomain[1])
    t_bc_top = self.generate_uniform_T_samples(num_sample - num_sample_per_edge)
    ids_bc_top = np.arange(x_bc_top.shape[0]) + counter
    counter += ids_bc_top.size

    ids_bc = np.concatenate([ids_bc_left, ids_bc_right, ids_bc_front, ids_bc_back, ids_bc_bottom, ids_bc_top])
    ids_all = np.concatenate([ids_dom, ids_ic, ids_bc])

    ids = {
        'domain': ids_dom, 
        'bc-left': ids_bc_left, 
        'bc-right': ids_bc_right,
        'bc-front': ids_bc_front,
        'bc-back': ids_bc_back,
        'bc-bottom': ids_bc_bottom,
        'bc-top': ids_bc_top,
        'ic': ids_ic,
        'bc': ids_bc, 
        'all': ids_all
    }

    assert all([t in ids.keys() for t in self.targets]), \
      'accepted target types: {}'.format(ids.keys())

    input_data = [
        np.concatenate([x_dom, x_ic, x_bc_left, x_bc_right, x_bc_front, x_bc_back, x_bc_bottom, x_bc_top]).reshape(-1,1),
        np.concatenate([y_dom, y_ic, y_bc_left, y_bc_right, y_bc_front, y_bc_back, y_bc_bottom, y_bc_top]).reshape(-1,1),
        np.concatenate([z_dom, z_ic, z_bc_left, z_bc_right, z_bc_front, z_bc_back, z_bc_bottom, z_bc_top]).reshape(-1,1),
        np.concatenate([t_dom, t_ic, t_bc_left, t_bc_right, t_bc_front, t_bc_back, t_bc_bottom, t_bc_top]).reshape(-1,1),
    ]
    total_sample = input_data[0].shape[0]

    target_data = []
    for i, tp in enumerate(self.targets):
      target_data.append(
          (ids[tp], 'zeros')
      )
      
    return input_data, target_data

  def get_test_grid(self, Nx=30, Ny=30, Nz=30, Nt=100):
    xs = np.linspace(self.Xdomain[0], self.Xdomain[1], Nx)
    ys = np.linspace(self.Ydomain[0], self.Ydomain[1], Ny)
    zs = np.linspace(self.Zdomain[0], self.Zdomain[1], Nz)
    if self.logT:
      ts = np.linspace(np.log1p(self.Tdomain[0]), np.log1p(self.Tdomain[1]), Nt)
      ts = np.exp(ts) - 1.0
    else:
      ts = np.linspace(self.Tdomain[0], self.Tdomain[1], Nt)
    return np.meshgrid(xs, ys, zs, ts)

  def plot_sample_batch(self, batch_size=500):
      ids = np.random.choice(len(self), batch_size, replace=False)
      x_data = self.input_data[0][ids,:]
      y_data = self.input_data[1][ids,:]
      z_data = self.input_data[2][ids,:]
      t_data = self.input_data[3][ids,:]
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      sc = ax.scatter(x_data, y_data, z_data, c=t_data, cmap='viridis')
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      plt.colorbar(sc, label='t')
      plt.title('Sample batch = {}'.format(batch_size))
      plt.show()

  def plot_data(self):
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      for t, (t_idx, t_val) in zip(self.targets, self.target_data):
        x_data = self.input_data[0][t_idx,:]
        y_data = self.input_data[1][t_idx,:]
        z_data = self.input_data[2][t_idx,:]
        t_data = self.input_data[3][t_idx,:]
        sc = ax.scatter(x_data, y_data, z_data, c=t_data, label=t, cmap='viridis')
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      plt.colorbar(sc, label='t')
      plt.legend(title="Training Data", bbox_to_anchor=(1.01, 1), loc='right', fontsize='x-small')
      fig.tight_layout()
      plt.show()