import numpy as np
import tensorflow as tf
import sys

def zero(x):
    return x-x

def one(x):
    return x-x+1

def two(x):
    return x-x+2

def three(x):
    return x - x + 3

def softplus0(x):
    return log_1_e_x(0.0)


def quarter(x):
    return x-x+0.25

def half(x):
    return x-x+0.5

def threeq(x):
    return x-x+0.75


def x(x):
    return x

def x_2(x):
    return tf.math.pow(x, 2)

def x_3(x):
    return tf.math.pow(x, 3)

def sqrt(x):
    # return tf.cast(tf.math.sqrt(x), dtype=float)
    return tf.math.sqrt(x)

def exp(x):
    # return tf.cast(tf.math.exp(x), dtype=tf.float32)
    return tf.math.exp(x)

def e_n_x_2(x):
    # return tf.cast(tf.math.exp(-1 * np.power(x, 2)), dtype=tf.float32)
    return tf.math.exp(-1 * x**2)
    #return np.power(math.e, -1 * np.power(x, 2))

def log_1_e_x(x):
    return tf.math.log(1 + tf.math.exp(x))

def log(x):
    return tf.math.log(tf.math.abs(x) + sys.float_info.epsilon)

def max0(x):
    return tf.math.maximum(0.0, x)

def max0_2(x):
    return tf.math.maximum(0.0, x)**2



def min0(x):
    return tf.math.minimum(0.0, x)

def dev(x1, x2):
    return x1/(x2 + sys.float_info.epsilon)

def sinc(x):
    x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
    return tf.sin(x) / x

def leaky_relu2(x):
    neg = tf.cast(x<0.0, tf.float32)
    return ((neg * 1.5 * x) + ((1.0-neg) * x**2))

def leaky_relu(x):
    neg = tf.cast(x<0.0, tf.float32)
    return ((neg * 1.5 * x) + ((1.0-neg) * x))


operators = {
    1: tf.math.add,
    2: tf.math.subtract,
    3: tf.math.multiply,
    4: tf.math.divide_no_nan,
    5: tf.math.maximum,
    6: tf.math.minimum,
    7: zero,
    8: one,
    9: tf.identity,
    10: tf.math.negative,
    11: tf.math.abs,
    12: x_2,
    13: x_3,
    14: tf.math.sqrt,
    15: exp,
    16: e_n_x_2,
    17: log_1_e_x,
    18: log,
    19: tf.math.sin,
    20: tf.math.sinh,
    21: tf.math.asinh,
    22: tf.math.cos,
    23: tf.math.cosh,
    24: tf.math.tanh,
    25: tf.math.atanh,
    26: tf.math.sigmoid,
    27: tf.math.erf,
    28: sinc,
    29: max0,
    30: min0
    # 31: softplus0,
    # 32: quarter,
    # 33: half,
    # 34: threeq,
    # 35: max0_2,
    # 36: two,
    # 37: three
    # 31: leaky_relu,
    # 32: leaky_relu2
}



class Function():
    def __init__(self, operators):
        self.operators = operators

    def s1(self, x):
        return BinaryOperator(self.operators[0],
                              UnaryOperator(self.operators[1], x),
                              UnaryOperator(self.operators[2], x)).calculate()

    def s2(self, x):
        return BinaryOperator(self.operators[0],
                              UnaryOperator(self.operators[1],
                                            BinaryOperator(self.operators[3],
                                                           UnaryOperator(self.operators[5], x),
                                                           UnaryOperator(self.operators[6], x))),
                              UnaryOperator(self.operators[2],
                                            BinaryOperator(self.operators[4],
                                                           UnaryOperator(self.operators[7], x),
                                                           UnaryOperator(self.operators[8], x)))).calculate()

    def sl(self, x):
        return UnaryOperator(self.operators[1],
                      BinaryOperator(self.operators[3],
                                     UnaryOperator(self.operators[5], x),
                                     UnaryOperator(self.operators[6], x))).calculate()

    def sr(self, x):
        return UnaryOperator(self.operators[2],
                      BinaryOperator(self.operators[4],
                                     UnaryOperator(self.operators[7], x),
                                     UnaryOperator(self.operators[8], x))).calculate()


class BinaryOperator():
    def __init__(self, operator, x1, x2):
        self.operator = operators[operator]
        self.x1 = x1
        self.x2 = x2

    def calculate(self):
        if ((isinstance(self.x1, UnaryOperator) or isinstance(self.x1, BinaryOperator)) and
                (isinstance(self.x2, UnaryOperator) or isinstance(self.x2, BinaryOperator))):
            return self.operator(self.x1.calculate(), self.x2.calculate())
        elif ((isinstance(self.x1, UnaryOperator) or isinstance(self.x1, BinaryOperator)) and
              (not isinstance(self.x2, UnaryOperator) and not isinstance(self.x2, BinaryOperator))):
            return self.operator(self.x1.calculate(), self.x2)
        elif ((not isinstance(self.x1, UnaryOperator) and not isinstance(self.x1, BinaryOperator)) and
              (isinstance(self.x2, UnaryOperator) or isinstance(self.x2, BinaryOperator))):
            return self.operator(self.x1, self.x2.calculate())
        else:
            return self.operator(self.x1, self.x2)


class UnaryOperator():
    def __init__(self, operator, x1):
        self.operator = operators[operator]
        self.x1 = x1

    def calculate(self):
        if (isinstance(self.x1, UnaryOperator) or isinstance(self.x1, BinaryOperator)):
            return self.operator(self.x1.calculate())
        else:
            return self.operator(self.x1)
