"""Likelihood-trained, causal GAS dynamics for reported-grid magnitude models."""
import math

import numpy as np

LN10 = math.log(10.)
B_MIN, B_MAX = .2, 2.


def sigmoid(value):
    if value >= 0:
        return 1. / (1. + math.exp(-min(value, 745.)))
    exponential = math.exp(max(value, -745.))
    return exponential / (1. + exponential)


def gr_statistics(state, excess, step, width):
    """Exact bin log probability, score and score derivative in the raw b state."""
    probability = sigmoid(state)
    b = B_MIN + (B_MAX-B_MIN)*probability
    rate = LN10*b
    rate_first = LN10*(B_MAX-B_MIN)*probability*(1.-probability)
    rate_second = rate_first*(1.-2.*probability)
    fine = math.expm1(rate*step)
    coarse = math.expm1(rate*width)
    mean = step/fine-width/coarse
    variance = step*step*(fine+1.)/(fine*fine)-width*width*(coarse+1.)/(coarse*coarse)
    logp = -rate*excess + math.log(-math.expm1(-rate*step))-math.log(-math.expm1(-rate*width))
    score = rate_first*(mean-excess)
    curvature = -variance*rate_first*rate_first+(mean-excess)*rate_second
    return logp, score, curvature, b


def mixture_statistics(state, first, second):
    probability = sigmoid(state)
    if state >= 0:
        normalizer = math.log1p(math.exp(-min(state, 745.)))
        log_first, log_second = -normalizer, -state-normalizer
    else:
        normalizer = math.log1p(math.exp(max(state, -745.)))
        log_first, log_second = state-normalizer, -normalizer
    left, right = log_first+first, log_second+second
    largest = max(left, right)
    logp = largest+math.log(math.exp(left-largest)+math.exp(right-largest))
    responsibility = sigmoid(state+first-second)
    score = responsibility-probability
    curvature = responsibility*(1.-responsibility)-probability*(1.-probability)
    return logp, score, curvature, probability


def parameters(theta, family):
    mean, log_half_life, log_gain = map(float, theta)
    half_life = math.exp(log_half_life)
    persistence = math.exp(-math.log(2.)/half_life)
    gain = math.exp(log_gain)
    alpha = gain if family == 'gr_free' else (1.-persistence)*gain
    return mean, persistence, alpha, half_life, gain


def contraction_bounds(theta, family, width):
    _, persistence, alpha, _, _ = parameters(theta, family)
    if family == 'mixture_contractive':
        lower, upper = persistence-.25*alpha, persistence+.25*alpha
    else:
        first = LN10*(B_MAX-B_MIN)/4.
        second = LN10*(B_MAX-B_MIN)/(6.*math.sqrt(3.))
        lower = persistence-alpha*(first*first*width*width/4.+second*width)
        upper = persistence+alpha*second*width
    return lower, upper


def filter_gas(theta, family, excess, step, width, mask=None, components=None,
               initial=None, return_trace=False):
    """Predict mark i before incorporating it; initial is independent of all marks."""
    mean, persistence, alpha, _, _ = parameters(theta, family)
    state = mean if initial is None else float(initial)
    total, count = 0., 0
    mixture = family == 'mixture_contractive'
    if mixture and (components is None or components.shape != (len(excess), 2)):
        raise ValueError('Two fixed component log probabilities are required.')
    if return_trace:
        logp = np.empty(len(excess));states = np.empty(len(excess))
        values = np.empty(len(excess));jacobian = np.empty(len(excess))
    for i in range(len(excess)):
        if mixture:
            lp, score, curvature, value = mixture_statistics(state, float(components[i,0]), float(components[i,1]))
        else:
            lp, score, curvature, value = gr_statistics(state, float(excess[i]), step, width)
        if mask is None or mask[i]:
            total -= lp;count += 1
        if return_trace:
            logp[i], states[i], values[i], jacobian[i] = lp, state, value, persistence+alpha*curvature
        state = mean+persistence*(state-mean)+alpha*score
    if return_trace:
        return dict(nll=total/max(1,count),logp=logp,state=states,value=values,
                    update_jacobian=jacobian,final_state=state)
    return total/max(1,count)
