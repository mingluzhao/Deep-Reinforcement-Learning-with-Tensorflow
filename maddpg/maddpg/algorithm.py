import tensorflow as tf
import numpy as np


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


class Actor:
    def __init__(self, actorModel):
        self.actorModel = actorModel


    def __call__(self, *args, **kwargs):




    def train(self):


    def act(self, stateBatch, noise = None):


    def getActionGradients:

