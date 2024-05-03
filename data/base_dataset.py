import abc
import os
from data.image_folder import make_dataset
#import torchvision.transforms as transforms

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



def get_transform(opt, params=None,grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list=[]
    if grayscale:
        transform_list.append(transforms.Gray)
    if 'resize' in opt.preprocess:
        osize=[opt.load_size,opt.load_size]
        transform_list.append(transforms.Resize(osize,method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda imag:__scale_width(img,opt.load_size,opt.crop_size,metod)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'],opt.crop_size)))
    if opt.preprocess=='none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4,method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if convert:
        transform_list+=[transforms.ToTensor()]
        if grayscale:
            transform_list+=[transforms.Normalize((0.5,),(0.5,))]
        else:
            transform_list+=[transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    return transforms.Compose(transform_list)

def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
                transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
                transforms.InterpolationMode.NEAREST: Image.NEAREST,
                transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]

def __make_power_2(img,base,method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    
    return img.resize((w, h), method)

def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

class UnalignedDataset(BaseDataset):

    def __init__(self,opt):
        BaseDataset.__init__(self,opt)
        self.dir_A=os.path.join(opt.dataroot,opt.phase+'A')
        self.dir_B=os.path.join(opt.dataroot,opt.phase+'B')

        self.A_paths=sorted(make_dataset(self.dir_A,opt.max_dataset_size))
        self.B_paths=sorted(make_dataset(self.dir_B,opt.max_dataset_size))
        self.A_size=len(self.A_paths)
        self.B_size=len(self.B_paths)
        btoA=self.opt.direction=='BtoA'
        input_nc=self.opt.output_nc if btoA else self.opt.input_nc
        output_nc=self.opt.input_nc if btoA else self.opt.output_nc
        
        self.transform_A=get_transform(self.opt,gray_scale=(input_nc==1))
        self.transform_B=get_transform(self.opt,gray_scale=(output_nc==1))
    
    def __get__item(self,index):
        A_path=self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B=index%self.B_size
        else:
            index_B=random.randint(0,self.B_size-1)
        B_path=self.B_paths[index_B]
        A_img=Image.open(A_path).convert('RGB')
        B_img=Image.open(B_path).convert('RGB')

        A=self.transform_A(A_img)
        B=self.transform_B(B_img)

        return {'A':A, 'B':B,'A_paths':A_path,'B_paths':B_path}
    
    def __len__(self):
        return max(self.A_size,self.B_size)
    
