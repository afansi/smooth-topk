import torch
import torch.nn as nn
import numpy as np
import topk.functional as F

from topk.utils import detect_large


class _SVMLoss(nn.Module):

    def __init__(self, n_classes, alpha):

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer('labels', torch.from_numpy(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y, weight=None, reduction='mean'):

        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        nn.Module.cpu()
        self.get_losses()
        return self
        
    def _handle_multi_dim(self, x, y):
        x_size = None
        if x.ndim > 2:
            x = x.transpose(1, -1)
            x_size = x.size()[:-1]
            x = x.reshape(-1, x.size(-1))
            y = y.reshape(-1)
        return x_size, x, y


class MaxTop1SVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None):

        super(MaxTop1SVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.get_losses()

    def forward(self, x, y, weight=None, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none')
        if weight is not None:
            assert weight.ndim == 1 and weight.size(0) == self.n_classes
        x_size, x, y = self._handle_multi_dim(x, y)
        output = self.F(x, y)
        if weight is not None:
            output = weight[y] * output
        if x_size is not None:
            output = output.reshape(*x_size)
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'sum:
            return output.sum()
        return output

    def get_losses(self):
        self.F = F.Top1_Hard_SVM(self.labels, self.alpha)


class MaxTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, k=5):

        super(MaxTopkSVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.k = k
        self.get_losses()

    def forward(self, x, y, weight=None, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none')
        if weight is not None:
            assert weight.ndim == 1 and weight.size(0) == self.n_classes
        x_size, x, y = self._handle_multi_dim(x, y)
        output = self.F(x, y)
        if weight is not None:
            output = weight[y] * output
        if x_size is not None:
            output = output.reshape(*x_size)
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'sum:
            return output.sum()
        return output

    def get_losses(self):
        self.F = F.Topk_Hard_SVM(self.labels, self.k, self.alpha)


class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.):
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y, weight=None, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none')
        if weight is not None:
            assert weight.ndim == 1 and weight.size(0) == self.n_classes
        x_size, x, y = self._handle_multi_dim(x, y)
        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        output = []
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            out = self.F_s(x_s, y_s)
            output.append(out)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            out = self.F_h(x_h, y_h)
            output.append(out)

        if len(output) > 1:
            res = torch.cat(output, dim=0)
            res[smooth] = output[0]
            res[hard] = output[1]
        else:
            res = output[0]

        output = res
        if weight is not None:
            output = weight[y] * output
        if x_size is not None:
            output = output.reshape(*x_size)
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'sum:
            return output.sum()
        return output

    def get_losses(self):
        self.F_h = F.Top1_Hard_SVM(self.labels, self.alpha)
        self.F_s = F.Top1_Smooth_SVM(self.labels, self.tau, self.alpha)


class SmoothTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, tau=1., k=5):
        super(SmoothTopkSVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.k = k
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y, weight=None, reduction='mean'):
        assert reduction in ('mean', 'sum', 'none')
        if weight is not None:
            assert weight.ndim == 1 and weight.size(0) == self.n_classes
        x_size, x, y = self._handle_multi_dim(x, y)
        smooth, hard = detect_large(x, self.k, self.tau, self.thresh)

        output = []
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            out = self.F_s(x_s, y_s)
            output.append(out)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            out = self.F_h(x_h, y_h)
            output.append(out)

        if len(output) > 1:
            res = torch.cat(output, dim=0)
            res[smooth] = output[0]
            res[hard] = output[1]
        else:
            res = output[0]

        output = res
        if weight is not None:
            output = weight[y] * output
        if x_size is not None:
            output = output.reshape(*x_size)
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'sum:
            return output.sum()
        return output

    def get_losses(self):
        self.F_h = F.Topk_Hard_SVM(self.labels, self.k, self.alpha)
        self.F_s = F.Topk_Smooth_SVM(self.labels, self.k, self.tau, self.alpha)
