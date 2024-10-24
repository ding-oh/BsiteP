import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, D, H, W = x.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(x)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        Args:
            num_channels (int): No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights=None):
        """
        Args:
            weights (torch.Tensor): weights for few shot learning
            x: X, shape = (batch_size, num_channels, D, H, W)

        Returns:
            (torch.Tensor): output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = x.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(x, weights)
        else:
            out = self.conv(x)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(x, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): No of input channels
            reduction_ratio (int): By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor

class ConvolutionBlock(nn.Module):
    def __init__(self, strides,channels):
        super().__init__()
        self.strides = strides
        
        self.channel1, self.channel2, self.channel3, self.channel4 = channels
        self.in_planes = self.channel2
        self.conv3d_r = nn.Conv3d(self.channel1,
                                  self.channel4,
                                  kernel_size=1,
                                  stride = self.strides)
        self.batch_normalization_r = nn.BatchNorm3d(self.in_planes)

        self.conv3d1 = nn.Conv3d(self.channel1,
                               self.channel2,
                               kernel_size=1,
                               stride= self.strides,
                               bias=False)
        self.batch_normalization1 = nn.BatchNorm3d(self.in_planes)
        self.activation1 = nn.ReLU()

        self.conv3d2 = nn.Conv3d(self.channel2,
                                 self.channel3,
                                 kernel_size=3,
                                 padding='same')
        self.batch_normalization2 = nn.BatchNorm3d(self.in_planes)
        self.activation2 = nn.ReLU()

        self.conv3d3 = nn.Conv3d(self.channel3,
                                 self.channel4,
                                 kernel_size=1)
        self.batch_normalization3 = nn.BatchNorm3d(self.in_planes)

        self.activation3 = nn.ReLU()


    def forward(self,x):
        r = self.conv3d_r(x)
        r = self.batch_normalization_r(r)

        x = self.conv3d1(x)
        x = self.batch_normalization1(x)
        x = self.activation1(x)

        x = self.conv3d2(x)
        x = self.batch_normalization2(x)
        x = self.activation2(x)

        x = self.conv3d3(x)
        x = self.batch_normalization3(x)

        x = x+r

        x = self.activation3(x)

        return x
    
class IdentityBlock(nn.Module):
    def __init__(self, channels,layer=None):
        super().__init__()
        
        self.channel1, self.channel2, self.channel3, self.channel4 = channels
        self.in_planes = self.channel2
        self.conv3d1 = nn.Conv3d(self.channel1,
                               self.channel2,
                               kernel_size=1,
                               bias=False)
        if layer==None:
            self.batch_normalization1 = nn.BatchNorm3d(self.in_planes)
        self.activation1 = nn.ReLU()

        self.conv3d2 = nn.Conv3d(self.channel2,
                                 self.channel3,
                                 kernel_size=3,
                                 padding='same')
        if layer==None:
            self.batch_normalization2 = nn.BatchNorm3d(self.in_planes)
        self.activation2 = nn.ReLU()

        self.conv3d3 = nn.Conv3d(self.channel3,
                                 self.channel4,
                                 kernel_size=1)
        if layer==None:
            self.batch_normalization3 = nn.BatchNorm3d(self.in_planes)

        self.activation3 = nn.ReLU()

    def forward(self,x):
        r  = x.clone()

        x = self.conv3d1(x)
        x = self.batch_normalization1(x)
        x = self.activation1(x)

        x = self.conv3d2(x)
        x = self.batch_normalization2(x)
        x = self.activation2(x)

        x = self.conv3d3(x)
        x = self.batch_normalization3(x)

        x = x+r

        x = self.activation3(x)

        return x
    
class UpSamplingBlock(nn.Module):
    def __init__(self,channels,stride,size,padding='same',layer=None):
        super().__init__()
        self.scale = size
       
        self.strides = stride
        self.channel1, self.channel2, self.channel3, self.channel4 = channels
        self.in_planes = self.channel2

        self.conv3d_r = nn.Conv3d(self.channel1,
                                  self.channel4,
                                  kernel_size=1,
                                  stride = self.strides,
                                  padding = 'same')
        if layer ==None:
            self.batch_normalization_r = nn.BatchNorm3d(self.in_planes)

        self.conv3d1 = nn.Conv3d(self.channel1,
                               self.channel2,
                               kernel_size=1,
                               stride = self.strides,
                               bias=False)
        if layer==None:
            self.batch_normalization1 = nn.BatchNorm3d(self.in_planes)
        self.activation1 = nn.ReLU()

        self.conv3d2 = nn.Conv3d(self.channel2,
                                 self.channel3,
                                 kernel_size=3,
                                 padding='same')
        if layer==None:
            self.batch_normalization2 = nn.BatchNorm3d(self.in_planes)
        self.activation2 = nn.ReLU()

        self.conv3d3 = nn.Conv3d(self.channel3,
                                 self.channel4,
                                 kernel_size=1)
        if layer==None:
            self.batch_normalization3 = nn.BatchNorm3d(self.in_planes)


        self.activation3 = nn.ReLU()

    def forward(self,x):
        r  = x.clone()
        x = F.interpolate(x,scale_factor=self.scale,mode = 'trilinear')
        x = self.conv3d1(x)
        x = self.batch_normalization1(x)
        x = self.activation1(x)

        x = self.conv3d2(x)
        x = self.batch_normalization2(x)
        x = self.activation2(x)

        x = self.conv3d3(x)
        x = self.batch_normalization3(x)

        r = F.interpolate(r,scale_factor=self.scale, mode = 'trilinear')
        r = self.conv3d_r(r)
        r = self.batch_normalization_r(r)

        x = x+r

        x = self.activation3(x)

        return x
        
class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_channels = 18
        f = input_channels
        
        ## downsampling
        self.Convblock1 = ConvolutionBlock(channels = [input_channels, f,f,f],strides = 1)
        self.identblock1 = IdentityBlock(channels = [f,f,f,f])
        self.skipidentityblock1 = IdentityBlock(channels = [f,f,f,f])
                               
        self.Convblock2 = ConvolutionBlock(channels = [f,f*2,f*2,f*2],strides = 2)
        self.identblock2 = IdentityBlock(channels = [f*2,f*2,f*2,f*2])
        self.skipidentityblock2 = IdentityBlock(channels = [f*2,f*2,f*2,f*2])

        self.Convblock3 = ConvolutionBlock(channels = [f*2, f*4,f*4,f*4],strides = 2)
        self.identblock3 = IdentityBlock(channels = [f*4,f*4,f*4,f*4])
        self.skipidentityblock3 = IdentityBlock(channels = [f*4,f*4,f*4,f*4])
                 
        self.Convblock4 = ConvolutionBlock(channels = [f*4, f*8,f*8,f*8],strides = 3)
        self.identblock4 = IdentityBlock(channels = [f*8,f*8,f*8,f*8])
        self.skipidentityblock4 = IdentityBlock(channels = [f*8,f*8,f*8,f*8])
                 
        self.Convblock5 = ConvolutionBlock(channels = [f*8, f*16,f*16,f*16],strides = 3)
        self.identblock5 = IdentityBlock(channels = [f*16,f*16,f*16,f*16])
                                
        ## upsampling
        self.upblock6 = UpSamplingBlock(channels = [f*16,f*16,f*16,f*16],size=3,stride=1)
        self.identblock6 = IdentityBlock(channels=[f*16,f*16,f*16,f*16])

        self.upblock7 = UpSamplingBlock(channels = [f*16+f*8,f*8,f*8,f*8],size=3,stride=1)
        self.identblock7 = IdentityBlock(channels=[f*8,f*8,f*8,f*8])

        self.upblock8 = UpSamplingBlock(channels = [f*8+f*4,f*4,f*4,f*4],size=2,stride=1)
        self.identblock8 = IdentityBlock(channels=[f*4,f*4,f*4,f*4])

        self.upblock9 = UpSamplingBlock(channels = [f*4+f*2,f*2,f*2,f*2],size=2,stride=1)
        self.identblock9 = IdentityBlock(channels=[f*2,f*2,f*2,f*2])

        self.finalconv10 = nn.Conv3d(f*2+f,1,kernel_size=1)
        self.finalact10 = nn.Sigmoid()


    def forward(self,x):
        skip = []
        x = self.Convblock1(x)
        x = self.identblock1(x)
        skip.append(self.skipidentityblock1(x))

        x = self.Convblock2(x)
        x = self.identblock2(x)
        skip.append(self.skipidentityblock2(x))
        
        
        x = self.Convblock3(x)
        x = self.identblock3(x)
        skip.append(self.skipidentityblock3(x))

        x = self.Convblock4(x)
        x = self.identblock4(x)
        skip.append(self.skipidentityblock4(x))

        x = self.Convblock5(x)
        x = self.identblock5(x)
        
        x = self.upblock6(x)
        x = self.identblock6(x)
        x = torch.cat([x,skip[-1]],dim=1)

        x = self.upblock7(x)
        x = self.identblock7(x)
        x = torch.cat([x,skip[-2]],dim=1)

        x = self.upblock8(x)
        x = self.identblock8(x)
        x = torch.cat([x,skip[-3]],dim=1)

        x = self.upblock9(x)
        x = self.identblock9(x)
        x = torch.cat([x,skip[-4]],dim=1)

        x = self.finalconv10(x)
        x = self.finalact10(x)

        return x
