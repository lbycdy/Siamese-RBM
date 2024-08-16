import torch
import torch.nn as nn
# from pysot.models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


def fuse_bn_sequential(block):
    """
    This function takes a sequential block and fuses the batch normalization with convolution

    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """
    # print("----------------------------",type(block))
    # if "ConvLayer" not in str(type(block)):
    #     if not isinstance(block, nn.Sequential):
    #         return block

    if not isinstance(block, nn.Sequential):
        return block
    # if len(block)>8:
    #     return block
    # print("----------------------------------------------------------------")
    # print("before:",block)
    stack = []
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d):
            if isinstance(stack[-1], nn.Conv2d) or isinstance(stack[-1], nn.ConvTranspose2d):
                bn_st_dict = m.state_dict()
                conv_st_dict = stack[-1].state_dict()

                # BatchNorm params
                eps = m.eps
                mu = bn_st_dict['running_mean']
                var = bn_st_dict['running_var']
                gamma = bn_st_dict['weight']

                if 'bias' in bn_st_dict:
                    beta = bn_st_dict['bias']
                else:
                    beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

                # Conv params
                W = conv_st_dict['weight']
                if isinstance(stack[-1], nn.ConvTranspose2d):
                    W = W.transpose(0,1)
                if 'bias' in conv_st_dict:
                    bias = conv_st_dict['bias']
                else:
                    bias = torch.zeros(W.size(0)).float().to(gamma.device)


                denom = torch.sqrt(var + eps)
                b = beta - gamma.mul(mu).div(denom)
                A = gamma.div(denom)
                bias *= A
                A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

                W.mul_(A)
                bias.add_(b)
                if isinstance(stack[-1], nn.ConvTranspose2d):
                    W = W.transpose(0,1)
                stack[-1].weight.data.copy_(W)
                if stack[-1].bias is None:
                    stack[-1].bias = torch.nn.Parameter(bias)
                else:
                    stack[-1].bias.data.copy_(bias)

        else:
            stack.append(m)
    # print("after:",stack)
    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
    return model
