import pytest
import numpy as np
import torch as th
import torch.autograd

from core.agent.evo import ConvolutionModel


kernel_sizes_test = ((3,), (5,), (3, 3), (3, 5), (3, 5, 3),)


def test_convolution_model_init():
    model = ConvolutionModel(num_act_channels=3,
                             num_obs_channels=3,
                             kernel_sizes=(3, 5),
                             p_agent_dropout=0.)

    def all_zeros(model):
        return all(th.all(k.weight == 0) for k in model.kernels)

    assert not all_zeros(model)


def prepare_raw_tensor(tensor):
    return th.reshape(tensor, (1, *tensor.shape))


@pytest.mark.parametrize('field_size', [(12, 12), (96, 96), (12, 8)])
@pytest.mark.parametrize('kernel_sizes', kernel_sizes_test)
def test_convolution_model_apply(field_size, kernel_sizes):
    num_obs_channels = 3
    num_act_channels = 2
    # NB: correct 4D shape for torch data, needed for valid padding
    obs_shape = (1, num_obs_channels, *field_size)
    input_data = th.rand(obs_shape)

    model = ConvolutionModel(num_act_channels=num_act_channels,
                             num_obs_channels=num_obs_channels,
                             kernel_sizes=kernel_sizes,
                             p_agent_dropout=0.)
    model.init_weights()
    output = model.forward(input_data)

    # Channels can differ, so use 0th channel
    assert th.any(input_data[0, 0, :, :] != output[0, 0, :, :])
    assert num_act_channels in output.shape


@pytest.mark.parametrize('field_size', [(12, 12), (12, 15)])
def test_grad(field_size):
    torch.autograd.set_detect_anomaly(True)

    num_obs_channels = 3
    num_act_channels = 2
    kernel_sizes = (3, 5)
    # NB: correct 4D shape for torch data, needed for valid padding
    obs_shape = (1, num_obs_channels, *field_size)
    input_data = th.rand(obs_shape, requires_grad=True)

    model = ConvolutionModel(num_act_channels=num_act_channels,
                             num_obs_channels=num_obs_channels,
                             kernel_sizes=kernel_sizes,
                             p_agent_dropout=0.25,
                             requires_grad=True)
    output_data = model.forward(input_data)
    output = output_data.mean()

    assert input_data.requires_grad
    assert output.requires_grad
    assert input_data.grad is None

    output.backward()

    assert input_data.grad is not None
