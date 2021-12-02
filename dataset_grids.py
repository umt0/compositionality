import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#this function generates ONE image
def generate_grid_random(dx,dy,nx=2,ny=2,img_size=32, point_size = 1):
     '''
     dx is the distance between points along x axis
     dy is the distance between points along y axis
     nx is the number of points along x axis
     ny is the number of points along y axis
     point_size is the size of the square points in the grid
     '''
    img = torch.zeros(img_size, img_size)
    
    #size of the grid
    size_grid_x = (dx+point_size)*nx
    size_grid_y = (dy+point_size)*ny
    if size_grid_x>32 or size_grid_y>32:
        print("error: too big grid!")
    else:        
        #generating the upper left point of grid
        #such that all grid is within the image
        bl0 = torch.randint(0, 1+img_size-size_grid_x, (1,))
        bl1 = torch.randint(0, 1+img_size-size_grid_y, (1,))

        #indices of grid points
        dxeff = dx+1
        dyeff =dy+1
        x_indx = [(bl0+jj*dxeff) for jj in range(nx)]
        y_indx = [(bl1+jj*dyeff) for jj in range(ny)]

        for xx in x_indx:
            img[xx,y_indx] = 1.0  
    return img
  

def create_dataset():
  '''
  this functions create the dataset
  OUTPUT: "dataset_grid/data_grid.pt" and "dataset_grid/lab_grid.pt"
  
  As it is now, the images created can have any combination of dx, dy and of nx and ny
  with constraints: (i) dx!=dy and (ii) nx and ny>2 (iii) it fits into img_size square image
  20 images are created for any possible tuple (dx, dy, nx, ny)
  
  if dy>dx: label +1
  otherwise: label -1
  
  as above, 54160 images are created
  '''
  img_size = 32                    
  point_size = 1
  dx_vec = range(1,1+img_size-2*point_size,1)
  dy_vec = range(1,1+img_size-2*point_size,1)

  N =0
  num_samples = 20
  for dx in dx_vec:
      for nx in range(2,1+int(img_size/(dx+1))):
          for dy in dy_vec:
              for ny in range(2,1+int(img_size/(dy+1))):
                  if dx!=dy:
                      for kk in range(num_samples):
                          N+=1
                          
  data = torch.empty((N,1,img_size,img_size))
  labs = torch.empty((N,1))                    

  count = 0
  for dx in dx_vec:
      for nx in range(2,1+int(img_size/(dx+1))):
          for dy in dy_vec:
              for ny in range(2,1+int(img_size/(dy+1))):
                  if dx!=dy:
                      for kk in range(num_samples):
                          image_sample = generate_grid_random(dx=dx,dy=dy,nx=nx,ny=ny)
                          lab = torch.tensor([2*int(dy-dx>0)-1])

                          data[count,0,:,:] = image_sample
                          labs[count,0] = lab
                          #print("==================================")
                          #print("okay dx: "+str(dx)+" nx: "+str(nx))
                          #print("okay dy: "+str(dy)+" ny: "+str(ny))
                          count+=1

                      #plt.savefig("dataset_grid/patt_dx_%d_nx_%d_dy_%d_ny_%d.png"%(dx,nx,dy,ny))
                      #torch.save(lab,"dataset_grid/lab_patt_dx_%d_nx_%d_dy_%d_ny_%d.txt"%(dx,nx,dy,ny))
  print(count)
  torch.save(data,"dataset_grid/data_grid.pt")
  torch.save(labs,"dataset_grid/lab_grid.pt")
  
#given the .pt files, this function creates an iterator dataloader similar to the MNIST one
def create_dataloader():
  class GridsDataset(Dataset):
    """Grids dataset."""

    def __init__(self, data_file, lab_file, transform=None):
        """
        Args:
            data_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data_file
        self.label = lab_file
        #self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        '''
        sample = self.data[idx,0,:,:]
        lab_sample = self.label[idx,0]
        if self.transform:
            sample = self.transform(sample)

        return sample, lab_sample
      
  data_file = torch.load("dataset_grid/data_grid.pt")
  lab_file = torch.load("dataset_grid/lab_grid.pt")

  grid_dataset = GridsDataset(data_file,lab_file)
  dataloader = DataLoader(grid_dataset, batch_size=10, shuffle=True)
  dataloader = iter(dataloader)
  return dataloader
