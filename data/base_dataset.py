import abc

class BaseDataset(data.Dataset,ABC):
    
    def __init__(self,opt):
        self.opt=opt
        self.root=opt.dataroot
    
    @staticmethod
    def modfiy_commandline_options(parser,is_train):
        return parser
    
    @abstractmethod
    def __len__(self):
        return 0
    
    @abstractmethod
    def __getitem__(self,index):
        pass

def get_params(opt,size):
    w,h=size
    new_h=h
    new_w=w
    