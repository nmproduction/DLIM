import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import util.util as util
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn import init


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        try:
            subnorm_type
        except:
            subnorm_type = 'instance'

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        # elif subnorm_type == 'sync_batch':
        #     norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer

class SesameMultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='sesame_n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc = None):
        super().__init__()
        self.opt = opt
        opt.num_D = 2
        opt.netD_subarch = 'sesame_n_layer'
        opt.no_ganFeat_loss = False

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc = None):
        subarch = opt.netD_subarch
        if subarch == 'sesame_n_layer':
            netD = SesameNLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# Defines the SESAME discriminator with the specified arguments.
class SesameNLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc=None):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = 1
        nf = opt.ndf
        if input_nc is None:
            input_nc = opt.input_nc

        branch = []
        sizes = (input_nc - 3, 3) 
        original_nf = nf
        for input_nc in sizes: 
            nf = original_nf
            norm_layer = get_nonspade_norm_layer(opt)
            sequence = [[nn.Conv2d(opt.input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, False)]]

            for n in range(1, 3):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]

            branch.append(sequence)
            
        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[0][n]))
        self.sem_sequence = nn.Sequential(*sem_sequence)

        sequence = branch[1]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
	#self.model = nn.Sequential(*sequence)

	
        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            self.img_sequence.append(nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
            
        return input_nc

    def forward(self, input):
        img, sem = input[:,-3:], input[:,:-3]
        sem_results = self.sem_sequence(sem)
        results = [img]
        for submodel in self.img_sequence[:-1]:
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence[-1](intermediate_output))

        get_intermediate_features = False
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

    def my_dot(self, x, y):
        return x + x * y.sum(1).unsqueeze(1)



def define_D(opt):
    return create_network(opt)

def create_network(opt):
    net = SesameMultiscaleDiscriminator(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, 0.02)
    return net