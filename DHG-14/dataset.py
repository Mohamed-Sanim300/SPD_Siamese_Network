class Shreck(Dataset):

  def __init__(self, train=True,transform=None):

    self.transform = transform
    if train:
      self.data_idx=np.loadtxt("/content/drive/MyDrive/shreck2017/HandGestureDataset_SHREC2017/train_gestures.txt")
    else:
      self.data_idx=np.loadtxt("/content/drive/MyDrive/shreck2017/HandGestureDataset_SHREC2017/test_gestures.txt")
    self.full_path=["/content/drive/MyDrive/shreck2017/HandGestureDataset_SHREC2017/gesture_"+str(int(self.data_idx[i,0]))+"/finger_"+str(int(self.data_idx[i,1]))+"/subject_"+str(int(self.data_idx[i,2]))+"/essai_"+str(int(self.data_idx[i,3])) for i in range(np.shape(self.data_idx)[0])]
    #self.link=[int(self.data_idx[i,0]),int(self.data_idx[i,1]),int(self.data_idx[i,2]),int(self.data_idx[i,3])]
    self.coord = {path:np.loadtxt(path+"/skeletons_world.txt")[:,6:] for path in self.full_path }
    
    self.Imgs = {path:[i for i in os.listdir(path) if i.endswith(".png")] for path in self.full_path}
    

  def __len__(self):
    return len(self.coord)

  def __getitem__(self, idx):
    path=self.full_path[idx]
    data=self.coord[path]
    row,col=np.shape(data)
    data=torch.from_numpy(data)
    data=data.T.reshape((1,col,row))
    data=nn.functional.interpolate(data,size=500).squeeze().T
    data=vg.normalize(data)
    data=data.reshape((data.size(0),20,3,1))
    label=torch.tensor([int(self.data_idx[idx][0]-1)])
    #label=torch.zeros(14)
    #label[int(self.data_idx[idx][0]-1)]=1
    sample={"data":data,"label":label}
    return sample
