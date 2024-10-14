
#--------------------------------
# Initialize: Custom dataset
#--------------------------------

class customDataset(Dataset):
    def __init__(self, data, transform_sample, transform_target):
        logger.debug("Initializing customDataset")
        self.hologram = torch.tensor(data)
        shape = self.hologram.shape
        self.hologram = self.hologram.view(1,1,shape[0],shape[1])
        self.transform_sample = transform_sample
        self.transform_target = transform_target

        self.sample = torch.ones(1,shape[0], shape[1])

    def __len__(self):
        return len(self.hologram)

    def __getitem__(self, idx):

        target = self.hologram[idx]
        
        if self.transform_sample is not None and self.transform_target is not None:
            target = self.transform_target(target)
            sample = self.transform_sample(self.sample).to(torch.complex64)
        else:
            sample = self.sample

        slm_sample = torch.abs(1-torch.abs(target))
        slm_sample = (slm_sample * 255).to(torch.uint8)

        #target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=10)

        return sample, slm_sample, target


class Hologram_DataModule(LightningDataModule):
    def __init__(self, params: dict, transform:str = "") -> None:
        super().__init__() 
        logger.debug("Initializing Wavefront_MNIST_DataModule")
        self.params = params.copy()
        self.Nx = self.params['Nxp']
        self.Ny = self.params['Nyp']
        self.n_cpus = self.params['n_cpus']
        self.path_data = self.params['paths']['path_data']
        self.path_root = self.params['paths']['path_root']
        self.path_data = os.path.join(self.path_root,self.path_data)
        logger.debug("Setting path_data to {}".format(self.path_data))

        self.initialize_transform()
        self.initialize_cpus(self.n_cpus)

    def initialize_transform(self) -> None:
        resize_row = self.params['resize_row']
        resize_col = self.params['resize_col']

        self.sample_transform = transforms.Compose([
                transforms.Resize((self.Nx, self.Ny), antialias=True), # type: ignore
                ct.Threshold(0.2),
                ct.WavefrontTransform(self.params['wavefront_transform'])])


        pad_x_left = pad_x_right = int(torch.div((self.Nx - resize_row), 2, rounding_mode='floor'))
        pad_y_left = pad_y_right = int(torch.div((self.Ny - resize_col), 2, rounding_mode='floor'))

        padding = (pad_y_left, pad_x_left, pad_y_right, pad_x_right)

        self.target_transform = transforms.Compose([
                transforms.Resize((resize_row, resize_col), antialias=True), # type: ignore
                transforms.Pad(padding),
                ct.Threshold(0.2),
                ct.WavefrontTransform(self.params['wavefront_transform'])])

    def initialize_cpus(self, n_cpus:int) -> None:
        # Make sure default number of cpus is not more than the system has
        if n_cpus > os.cpu_count(): # type: ignore
            n_cpus = 1
        self.n_cpus = n_cpus 
        logger.debug("Setting CPUS to {}".format(self.n_cpus))

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        
        hologram = np.load(os.path.join(self.path_data, 'miz_test.npy'))

        if stage == "fit" or stage is None:
            self.hologram_train = customDataset(hologram, self.sample_transform, self.target_transform)

    def train_dataloader(self):
        return DataLoader(self.hologram_train,
                          batch_size=1,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.hologram_train,
                          batch_size=1,
                          num_workers=self.n_cpus,
                          persistent_workers=True,
                          shuffle=False)

