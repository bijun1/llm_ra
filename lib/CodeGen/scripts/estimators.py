import torch

@torch.no_grad()
def conditional_gumbel(logits, D, k=1):
    # iid. exponential
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D * (-torch.log(Ei) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted - logits

def replace_gradient(value, surrogate):
    """Returns `value` but backpropagates gradients through `surrogate`."""
    return surrogate + (value - surrogate).detach()

def gumbel_rao(logits, k, temp=1.0, I=None):
    num_classes = logits.shape[-1]
    if I is None:
        I = torch.distributions.categorical.Categorical(logits=logits).sample()
    D = torch.nn.functional.one_hot(I, num_classes).float()
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = torch.nn.functional.softmax(adjusted/temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)


def gumbel_softmax(logits, tau=1, dim=-1):
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels
    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, y_soft

class ReinMaxCore(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        logits: torch.Tensor, 
        tau: torch.Tensor,
    ):
        y_soft = logits.softmax(dim=-1)
        sample = torch.multinomial(
            y_soft,
            num_samples=1,
            replacement=True,
        )
        one_hot_sample = torch.zeros_like(
            y_soft, 
            memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, sample, 1.0)
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau)
        return one_hot_sample, y_soft

    @staticmethod
    def backward(
        ctx, 
        grad_at_sample: torch.Tensor, 
        grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors
        
        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        
        grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        
        grad_at_input = grad_at_input_0 + grad_at_input_1
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None

def reinmax(
        logits: torch.Tensor, 
        tau: float, 
    ):
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMaxCore.apply(logits, logits.new_empty(1).fill_(tau))
    return grad_sample.view(shape)

def ST(
        logits: torch.Tensor, 
        tau: float, 
):
    y_hard = one_hot_multinomial(logits.softmax()) 
    y_soft_tau = (logits/tau).softmax()
    y_hard = y_soft_tau - y_soft_tau.detach() + y_hard 
    return y_hard