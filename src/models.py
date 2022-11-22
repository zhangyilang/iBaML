import torch
import torch.nn as nn
from collections import OrderedDict
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear
from src.container import MetaParameterDict


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class FourBlkCNN(MetaModule):
    def __init__(self, num_classes, in_channels=3, hidden_size=32, num_feat=800):
        super(FourBlkCNN, self).__init__()
        self.in_channels = in_channels
        self.out_features = num_classes

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            nn.Flatten()
        )

        self.classifier = MetaLinear(num_feat, num_classes)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits


class ReparamMetaModule(MetaModule):
    def __init__(self, model, num_samples_train=5, num_samples_eval=10):
        super(ReparamMetaModule, self).__init__()
        self.model = model
        self.num_samples_train = num_samples_train
        self.num_samples_eval = num_samples_eval

        # prior params
        self.mean = OrderedDict(model.meta_named_parameters())
        self.log_var = MetaParameterDict({'{0:02d}'.format(i): nn.Parameter(torch.rand_like(v)-10)  # init std ~ 0.007
                                                for i, v in enumerate(self.mean.values())})

        # posterior params (observing support data)
        self.post_mean = None
        self.post_var = None

    def forward(self, inputs, params=None):
        if params is None:
            mean = self.mean.values()
            log_var = self.log_var.values()
        else:
            mean = self.get_subdict(params, 'model').values()
            log_var = self.get_subdict(params, 'log_var').values()

        outputs = []
        num_samples = self.num_samples_train if self.training else self.num_samples_eval

        for idx_sample in range(num_samples):
            params_sample = OrderedDict(self.model.meta_named_parameters())
            for k, m, log_v in zip(params_sample.keys(), mean, log_var):
                params_sample[k] = m + torch.randn_like(m) * torch.exp(log_v * 0.5)
            outputs.append(self.model(inputs, params=params_sample))

        return outputs


if __name__ == '__main__':
    cnn = FourBlkCNN(1, 2)
    cnn_vi = ReparamMetaModule(cnn)
    print(OrderedDict(cnn.meta_named_parameters()).keys())
    print(cnn_vi.mean.keys())
    print(cnn_vi.log_var.keys())
    print(OrderedDict(cnn_vi.meta_named_parameters()).keys())
